import argparse
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path

try:
    import bpy
    from mathutils import Matrix
    from mathutils import Vector
except ImportError as exc:
    raise ImportError(
        'The synthetic renderer requires bpy when running with renderer_backend=python_bpy. '
        'Use .venv-render with the synthetic extra, or run through Blender with '
        'renderer_backend=blender_executable.'
    ) from exc


SUPPORTED_MODEL_SUFFIXES = {'.blend', '.blend1', '.fbx', '.obj', '.glb', '.gltf'}
ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'


def _parse_args():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(description='Render one diagnostic image from a 3D model.')
    parser.add_argument('--model-path')
    parser.add_argument('--output-path')
    parser.add_argument('--batch-plan')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--render-engine', default='BLENDER_EEVEE')
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--view-angle-degrees', type=float, default=0.0)
    parser.add_argument('--front-angle-degrees', type=float, default=0.0)
    parser.add_argument('--pitch-degrees', type=float, default=72.0)
    parser.add_argument('--roll-degrees', type=float, default=0.0)
    parser.add_argument('--camera-radius', type=float, default=4.0)
    parser.add_argument('--camera-lens', type=float, default=55.0)
    parser.add_argument('--light-energy', type=float, default=700.0)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--vertical-offset', type=float, default=0.0)
    parser.add_argument('--inspect-only', action='store_true')
    return parser.parse_args(argv)


def _clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def _append_blend_objects(blend_path):
    _clear_scene()
    with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)

    linked_count = 0
    for obj in data_to.objects:
        if obj is None:
            continue
        bpy.context.collection.objects.link(obj)
        linked_count += 1
    if linked_count == 0:
        raise RuntimeError(f'No objects could be loaded from {blend_path}.')


def _resolve_blend_path(model_path):
    header = model_path.read_bytes()[:8]
    if header.startswith(b'BLENDER'):
        return model_path, None
    if header.startswith(ZSTD_MAGIC):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                f'{model_path} appears to be Zstandard-compressed. Install zstandard in the render environment.'
            ) from exc
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.blend')
        temp_path = Path(temp_file.name)
        temp_file.close()
        try:
            with model_path.open('rb') as source, temp_path.open('wb') as target:
                zstd.ZstdDecompressor().copy_stream(source, target)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        if not temp_path.read_bytes()[:8].startswith(b'BLENDER'):
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f'Decompressed file from {model_path} is not a valid Blender file.')
        return temp_path, temp_path
    return model_path, None


def _load_model(model_path):
    suffix = model_path.suffix.lower()
    if suffix not in SUPPORTED_MODEL_SUFFIXES:
        raise ValueError(f'Unsupported model extension: {suffix}')

    if suffix in {'.blend', '.blend1'}:
        blend_path, temp_path = _resolve_blend_path(model_path)
        try:
            _append_blend_objects(blend_path)
        finally:
            if temp_path:
                temp_path.unlink(missing_ok=True)
    elif suffix == '.fbx':
        _clear_scene()
        bpy.ops.import_scene.fbx(filepath=str(model_path))
    elif suffix == '.obj':
        _clear_scene()
        if hasattr(bpy.ops.wm, 'obj_import'):
            bpy.ops.wm.obj_import(filepath=str(model_path))
        else:
            bpy.ops.import_scene.obj(filepath=str(model_path))
    elif suffix in {'.glb', '.gltf'}:
        _clear_scene()
        bpy.ops.import_scene.gltf(filepath=str(model_path))


def _mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']


def _inspect_scene(model_path):
    objects = list(bpy.context.scene.objects)
    mesh_objects = [obj for obj in objects if obj.type == 'MESH']
    armatures = [obj for obj in objects if obj.type == 'ARMATURE']
    constraints = [constraint for obj in objects for constraint in obj.constraints]
    shape_key_count = 0
    shape_key_objects = []
    parented_objects = []

    for obj in objects:
        if obj.parent is not None:
            parented_objects.append(obj.name)
        if obj.type == 'MESH' and obj.data and getattr(obj.data, 'shape_keys', None):
            keys = obj.data.shape_keys.key_blocks
            if keys:
                shape_key_count += len(keys)
                shape_key_objects.append(obj.name)

    bone_count = 0
    for armature in armatures:
        if armature.data and hasattr(armature.data, 'bones'):
            bone_count += len(armature.data.bones)

    action_names = sorted({action.name for action in bpy.data.actions})
    non_model_objects = [
        obj.name for obj in objects
        if obj.type not in {'MESH', 'ARMATURE', 'EMPTY'}
    ]
    movable_parts = sorted({
        *[obj.name for obj in armatures],
        *shape_key_objects,
        *[obj.name for obj in objects if obj.constraints],
        *parented_objects,
    })

    return {
        'model_path': str(model_path),
        'object_count': len(objects),
        'mesh_count': len(mesh_objects),
        'mesh_names': [obj.name for obj in mesh_objects],
        'armature_count': len(armatures),
        'armature_names': [obj.name for obj in armatures],
        'bone_count': bone_count,
        'has_armature': bool(armatures),
        'constraint_count': len(constraints),
        'has_constraints': bool(constraints),
        'has_shape_keys': shape_key_count > 0,
        'shape_key_count': shape_key_count,
        'action_count': len(action_names),
        'action_names': action_names,
        'parented_object_count': len(parented_objects),
        'movable_parts': movable_parts,
        'embedded_props': non_model_objects,
        'canonical_pose_required': bool(armatures or action_names or shape_key_count),
        'pose_variants_proposed': ['neutral'] if not (armatures or action_names or shape_key_count) else ['neutral', 'pose_review_required'],
    }


def _scene_bounds(objects):
    min_corner = Vector((math.inf, math.inf, math.inf))
    max_corner = Vector((-math.inf, -math.inf, -math.inf))
    for obj in objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world_corner.x)
            min_corner.y = min(min_corner.y, world_corner.y)
            min_corner.z = min(min_corner.z, world_corner.z)
            max_corner.x = max(max_corner.x, world_corner.x)
            max_corner.y = max(max_corner.y, world_corner.y)
            max_corner.z = max(max_corner.z, world_corner.z)
    return min_corner, max_corner


def _normalize_model(scale=1.0, vertical_offset=0.0):
    objects = _mesh_objects()
    if not objects:
        raise RuntimeError('No mesh objects found in the scene.')

    min_corner, max_corner = _scene_bounds(objects)
    center = (min_corner + max_corner) / 2.0
    dimensions = max_corner - min_corner
    max_dimension = max(dimensions.x, dimensions.y, dimensions.z)
    scale_factor = (2.0 / max_dimension) * float(scale) if max_dimension > 0 else float(scale)

    root = bpy.data.objects.new('dataset_model_root', None)
    bpy.context.collection.objects.link(root)
    root.location = (0, 0, vertical_offset)

    transform = (
        Matrix.Translation(Vector((0.0, 0.0, float(vertical_offset))))
        @ Matrix.Scale(scale_factor, 4)
        @ Matrix.Translation(-center)
    )
    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world
        obj.parent = root
        obj.matrix_parent_inverse = root.matrix_world.inverted()
    return root, objects, dimensions


def _look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def _create_camera(lens):
    camera_data = bpy.data.cameras.new('dataset_camera')
    camera = bpy.data.objects.new('dataset_camera', camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.data.lens = lens
    return camera


def _create_lights(energy):
    positions = [(-3.0, -4.0, 5.0), (4.0, 3.0, 4.0)]
    for idx, location in enumerate(positions):
        light_data = bpy.data.lights.new(f'dataset_area_light_{idx}', type='AREA')
        light_obj = bpy.data.objects.new(f'dataset_area_light_{idx}', light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = location
        light_data.size = 4.0
        light_data.energy = energy if idx == 0 else energy * 0.45


def _available_render_engines(scene):
    return {item.identifier for item in scene.render.bl_rna.properties['engine'].enum_items}


def _resolve_render_engine(scene, requested):
    requested = requested or 'BLENDER_EEVEE'
    available = _available_render_engines(scene)
    aliases = {
        'EEVEE': ['BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'],
        'BLENDER_EEVEE': ['BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'],
        'BLENDER_EEVEE_NEXT': ['BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'],
        'CYCLES': ['CYCLES', 'BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'],
    }
    for candidate in aliases.get(requested, [requested]):
        if candidate in available:
            if candidate != requested:
                print(f'Warning: render engine {requested} unavailable. Falling back to {candidate}.', file=sys.stderr)
            return candidate
    raise ValueError(f'Unsupported render engine {requested!r}. Available engines: {sorted(available)}')


def _configure_cycles_device(device):
    if device.upper() != 'GPU':
        return {'requested_device': device, 'cycles_device': 'CPU', 'compute_device_type': 'NONE', 'devices': []}

    preferences = bpy.context.preferences.addons['cycles'].preferences
    selected_type = None
    for compute_type in ('OPTIX', 'CUDA'):
        try:
            preferences.compute_device_type = compute_type
            preferences.refresh_devices()
        except Exception:
            continue
        gpu_devices = [dev for dev in preferences.devices if dev.type == compute_type]
        if gpu_devices:
            for dev in preferences.devices:
                dev.use = dev.type == compute_type
            selected_type = compute_type
            break

    if not selected_type:
        raise RuntimeError('Cycles GPU requested, but no OPTIX or CUDA device is available to bpy.')

    return {
        'requested_device': device,
        'cycles_device': 'GPU',
        'compute_device_type': selected_type,
        'devices': [
            {'name': dev.name, 'type': dev.type, 'use': bool(dev.use)}
            for dev in preferences.devices
        ],
    }


def _setup_render(image_size, render_engine, device):
    scene = bpy.context.scene
    scene.render.resolution_x = int(image_size)
    scene.render.resolution_y = int(image_size)
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    if str(render_engine).upper() == 'CYCLES':
        try:
            bpy.ops.preferences.addon_enable(module='cycles')
        except Exception:
            pass
    resolved_engine = _resolve_render_engine(scene, render_engine)
    scene.render.engine = resolved_engine

    if resolved_engine == 'CYCLES':
        scene.cycles.samples = 32
        scene.cycles.use_denoising = True
        device_metadata = _configure_cycles_device(device)
        scene.cycles.device = device_metadata['cycles_device']
    elif resolved_engine in {'BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'} and hasattr(scene, 'eevee'):
        scene.eevee.taa_render_samples = 32
        device_metadata = {'requested_device': device, 'cycles_device': None, 'compute_device_type': None, 'devices': []}
    else:
        device_metadata = {'requested_device': device, 'cycles_device': None, 'compute_device_type': None, 'devices': []}
    return resolved_engine, device_metadata


def _set_camera_pose(camera, view_angle_degrees, front_angle_degrees, pitch_degrees, roll_degrees, radius):
    theta = math.radians(front_angle_degrees + view_angle_degrees)
    phi = math.radians(pitch_degrees)
    camera.location = (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )
    _look_at(camera, (0, 0, 0.05))
    camera.rotation_euler.rotate_axis('Z', math.radians(roll_degrees))


def _set_world_background(seed):
    rng = random.Random(seed)
    world = bpy.context.scene.world or bpy.data.worlds.new('dataset_world')
    bpy.context.scene.world = world
    value = rng.uniform(0.72, 0.90)
    world.color = (value, value, value)


def _render_current_scene(output_path):
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def _setup_scene_for_model(args):
    model_path = Path(args.model_path).resolve()
    _load_model(model_path)
    model_root, mesh_objects, dimensions = _normalize_model(args.scale, args.vertical_offset)
    model_root.rotation_euler = (0, 0, 0)
    camera = _create_camera(args.camera_lens)
    _create_lights(args.light_energy)
    resolved_engine, device_metadata = _setup_render(args.image_size, args.render_engine, args.device)
    return model_path, camera, mesh_objects, dimensions, resolved_engine, device_metadata


def _render_batch(args):
    plan_path = Path(args.batch_plan).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    args.model_path = plan.get('model_path') or args.model_path
    if not args.model_path:
        raise ValueError('Batch plan must provide model_path or --model-path must be set.')

    started_at = time.perf_counter()
    model_path, camera, mesh_objects, dimensions, resolved_engine, device_metadata = _setup_scene_for_model(args)
    results = []
    for frame in plan.get('frames', []):
        frame_seed = int(frame.get('seed', args.seed))
        _set_world_background(frame_seed)
        _set_camera_pose(
            camera,
            float(frame.get('view_angle_degrees', args.view_angle_degrees)),
            float(frame.get('front_angle_degrees', args.front_angle_degrees)),
            float(frame.get('pitch_degrees', args.pitch_degrees)),
            float(frame.get('roll_degrees', args.roll_degrees)),
            float(frame.get('camera_radius', args.camera_radius)),
        )
        frame_started_at = time.perf_counter()
        _render_current_scene(frame['output_path'])
        results.append({
            'output_path': frame['output_path'],
            'seed': frame_seed,
            'view_angle_degrees': frame.get('view_angle_degrees', args.view_angle_degrees),
            'front_angle_degrees': frame.get('front_angle_degrees', args.front_angle_degrees),
            'pitch_degrees': frame.get('pitch_degrees', args.pitch_degrees),
            'roll_degrees': frame.get('roll_degrees', args.roll_degrees),
            'camera_radius': frame.get('camera_radius', args.camera_radius),
            'elapsed_seconds': round(time.perf_counter() - frame_started_at, 3),
        })

    result = {
        'model_path': str(model_path),
        'render_engine': resolved_engine,
        'bpy_version': bpy.app.version_string,
        'device': device_metadata,
        'mesh_count': len(mesh_objects),
        'dimensions': [dimensions.x, dimensions.y, dimensions.z],
        'elapsed_seconds': round(time.perf_counter() - started_at, 3),
        'frames': results,
    }
    print(json.dumps(result, ensure_ascii=False))


def main():
    args = _parse_args()
    if args.batch_plan:
        _render_batch(args)
        return
    if not args.model_path or not args.output_path:
        raise ValueError('--model-path and --output-path are required unless --batch-plan is used.')
    started_at = time.perf_counter()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path).resolve()
    _load_model(model_path)
    if args.inspect_only:
        print(json.dumps(_inspect_scene(model_path), ensure_ascii=False))
        return

    model_root, mesh_objects, dimensions = _normalize_model(args.scale, args.vertical_offset)
    model_root.rotation_euler = (0, 0, 0)
    camera = _create_camera(args.camera_lens)
    _create_lights(args.light_energy)
    _set_world_background(args.seed)
    resolved_engine, device_metadata = _setup_render(args.image_size, args.render_engine, args.device)
    _set_camera_pose(
        camera,
        args.view_angle_degrees,
        args.front_angle_degrees,
        args.pitch_degrees,
        args.roll_degrees,
        args.camera_radius,
    )

    _render_current_scene(output_path)

    result = {
        'model_path': str(model_path),
        'output_path': str(output_path),
        'render_engine': resolved_engine,
        'bpy_version': bpy.app.version_string,
        'device': device_metadata,
        'mesh_count': len(mesh_objects),
        'dimensions': [dimensions.x, dimensions.y, dimensions.z],
        'elapsed_seconds': round(time.perf_counter() - started_at, 3),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
