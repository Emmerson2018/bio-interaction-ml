import argparse
import math
import random
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
except ImportError as exc:
    raise ImportError(
        'The synthetic renderer requires the "bpy" Python package. '
        'Install the project synthetic extra or install bpy in the active Python environment.'
    ) from exc


def _parse_args():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--class-name', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--camera-radius-min', type=float, default=3.0)
    parser.add_argument('--camera-radius-max', type=float, default=5.5)
    parser.add_argument('--render-engine', default='CYCLES')
    parser.add_argument('--device', default='CUDA')
    parser.add_argument('--view-strategy', default='random_360', choices=['random_360', 'structured', 'fixed'])
    parser.add_argument('--front-angle-degrees', type=float, default=0.0)
    parser.add_argument('--fixed-view-angle-degrees', type=float, default=None)
    parser.add_argument('--close-view-ratio', type=float, default=0.25)
    parser.add_argument('--far-view-ratio', type=float, default=0.20)
    parser.add_argument('--psychedelic-bg-prob', type=float, default=0.0)
    parser.add_argument('--light-count-min', type=int, default=2)
    parser.add_argument('--light-count-max', type=int, default=2)
    parser.add_argument('--occlusion-prob', type=float, default=0.0)
    parser.add_argument('--occlusion-count-max', type=int, default=0)
    parser.add_argument('--occlusion-coverage-max', type=float, default=0.0)
    parser.add_argument('--noise-prob', type=float, default=0.0)
    parser.add_argument('--noise-types', default='')
    parser.add_argument('--noise-intensity-min', type=float, default=0.0)
    parser.add_argument('--noise-intensity-max', type=float, default=0.0)
    return parser.parse_args(argv)


def _clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def _load_model(model_path):
    suffix = model_path.suffix.lower()
    if suffix in {'.blend', '.blend1'}:
        bpy.ops.wm.open_mainfile(filepath=str(model_path))
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
    else:
        raise ValueError(f'Unsupported model extension: {suffix}')


def _mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']


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


def _normalize_model():
    objects = _mesh_objects()
    if not objects:
        raise RuntimeError('No mesh objects found in the scene.')

    min_corner, max_corner = _scene_bounds(objects)
    center = (min_corner + max_corner) / 2.0
    max_dimension = max((max_corner - min_corner).x, (max_corner - min_corner).y, (max_corner - min_corner).z)
    scale_factor = 2.0 / max_dimension if max_dimension > 0 else 1.0

    empty = bpy.data.objects.new('dataset_model_root', None)
    bpy.context.collection.objects.link(empty)
    empty.location = (0, 0, 0)

    for obj in objects:
        obj.location -= center
        obj.scale *= scale_factor
        obj.parent = empty
    return empty


def _look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def _create_camera():
    camera_data = bpy.data.cameras.new('dataset_camera')
    camera = bpy.data.objects.new('dataset_camera', camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera.data.lens = 55
    return camera


def _create_lights(max_count):
    lights = []
    for idx in range(max_count):
        light_data = bpy.data.lights.new(f'dataset_light_{idx}', type='AREA')
        light_obj = bpy.data.objects.new(f'dataset_light_{idx}', light_data)
        bpy.context.collection.objects.link(light_obj)
        lights.append(light_obj)
    return lights

def _create_occluders(max_count):
    occluders = []
    for idx in range(max_count):
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.object
        obj.name = f'dataset_occluder_{idx}'
        mat = bpy.data.materials.new(f'{obj.name}_mat')
        mat.use_nodes = True
        obj.data.materials.append(mat)
        occluders.append(obj)
    return occluders


def _resolve_render_engine(scene, render_engine):
    if render_engine == 'CYCLES':
        try:
            bpy.ops.preferences.addon_enable(module='cycles')
        except Exception:
            pass

    available_engines = {item.identifier for item in scene.render.bl_rna.properties['engine'].enum_items}
    if render_engine in available_engines:
        return render_engine

    aliases = {
        'BLENDER_EEVEE_NEXT': 'BLENDER_EEVEE',
        'EEVEE': 'BLENDER_EEVEE',
    }
    aliased_engine = aliases.get(render_engine)
    if aliased_engine in available_engines:
        return aliased_engine

    if render_engine == 'CYCLES' and 'BLENDER_EEVEE' in available_engines:
        print('Warning: CYCLES is not available in this bpy session. Falling back to BLENDER_EEVEE.', file=sys.stderr)
        return 'BLENDER_EEVEE'

    raise ValueError(f'Unsupported render engine: {render_engine}. Available engines: {sorted(available_engines)}')


def _setup_render(image_size, render_engine, device):
    scene = bpy.context.scene
    scene.render.resolution_x = image_size
    scene.render.resolution_y = image_size
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    render_engine = _resolve_render_engine(scene, render_engine)
    scene.render.engine = render_engine

    if render_engine == 'CYCLES':
        scene.cycles.samples = 64
        scene.cycles.use_denoising = True
        scene.cycles.device = 'CPU' if device == 'CPU' else 'GPU'
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = device
            for cycles_device in prefs.devices:
                cycles_device.use = device != 'CPU'
        except Exception:
            pass
    elif render_engine in {'BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'} and hasattr(scene, 'eevee'):
        scene.eevee.taa_render_samples = 64


def _randomize_materials(rng):
    for obj in _mesh_objects():
        if not obj.data.materials:
            material = bpy.data.materials.new(f'{obj.name}_dataset_material')
            obj.data.materials.append(material)
        for material in obj.data.materials:
            material.use_nodes = True
            bsdf = material.node_tree.nodes.get('Principled BSDF')
            if bsdf:
                base = rng.uniform(0.35, 0.85)
                color = (
                    min(1.0, base * rng.uniform(0.75, 1.25)),
                    min(1.0, base * rng.uniform(0.75, 1.25)),
                    min(1.0, base * rng.uniform(0.75, 1.25)),
                    1.0,
                )
                bsdf.inputs['Base Color'].default_value = color
                bsdf.inputs['Roughness'].default_value = rng.uniform(0.35, 0.9)


def _set_world_background(rng, psychedelic_prob=0.0):
    world = bpy.context.scene.world or bpy.data.worlds.new('dataset_world')
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    
    bg_node = nodes.new(type='ShaderNodeBackground')
    out_node = nodes.new(type='ShaderNodeOutputWorld')
    links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])
    
    if rng.random() < psychedelic_prob:
        tex_node = nodes.new(type='ShaderNodeTexVoronoi')
        tex_node.inputs['Scale'].default_value = rng.uniform(5.0, 50.0)
        ramp_node = nodes.new(type='ShaderNodeValToRGB')
        ramp_node.color_ramp.elements[0].color = (rng.random(), rng.random(), rng.random(), 1)
        ramp_node.color_ramp.elements[1].color = (rng.random(), rng.random(), rng.random(), 1)
        links.new(tex_node.outputs['Color'], ramp_node.inputs['Fac'])
        links.new(ramp_node.outputs['Color'], bg_node.inputs['Color'])
    else:
        bg_node.inputs['Color'].default_value = (
            rng.uniform(0.05, 0.95),
            rng.uniform(0.05, 0.95),
            rng.uniform(0.05, 0.95),
            1.0
        )


def _weighted_choice(rng, weighted_items):
    total = sum(weight for _, weight in weighted_items)
    threshold = rng.uniform(0, total)
    cumulative = 0.0
    for item, weight in weighted_items:
        cumulative += weight
        if threshold <= cumulative:
            return item
    return weighted_items[-1][0]


def _sample_view_angle(rng, strategy, front_angle_degrees, fixed_view_angle_degrees):
    if strategy == 'random_360':
        return rng.uniform(0, math.tau)

    if strategy == 'fixed':
        if fixed_view_angle_degrees is None:
            raise ValueError('fixed view strategy requires --fixed-view-angle-degrees.')
        return math.radians(fixed_view_angle_degrees)

    view_center, jitter = _weighted_choice(
        rng,
        [
            ((0, 14), 0.32),
            ((-35, 12), 0.27),
            ((35, 12), 0.27),
            ((-90, 10), 0.06),
            ((90, 10), 0.05),
            ((180, 14), 0.015),
            ((-180, 14), 0.015),
        ],
    )
    angle_degrees = front_angle_degrees + view_center + rng.uniform(-jitter, jitter)
    return math.radians(angle_degrees)


def _sample_camera_radius(rng, radius_min, radius_max, close_view_ratio, far_view_ratio):
    mode = _weighted_choice(
        rng,
        [
            ('close', max(0.0, close_view_ratio)),
            ('far', max(0.0, far_view_ratio)),
            ('medium', max(0.0, 1.0 - close_view_ratio - far_view_ratio)),
        ],
    )
    radius_range = radius_max - radius_min
    if mode == 'close':
        return rng.uniform(radius_min, radius_min + radius_range * 0.28)
    if mode == 'far':
        return rng.uniform(radius_min + radius_range * 0.70, radius_max)
    return rng.uniform(radius_min + radius_range * 0.25, radius_min + radius_range * 0.75)


def _randomize_occluders(rng, occluders, camera, prob, max_coverage):
    for obj in occluders:
        obj.hide_render = True
        
    if rng.random() > prob or not occluders:
        return
        
    count = rng.randint(1, len(occluders))
    for obj in occluders[:count]:
        obj.hide_render = False
        bsdf = obj.data.materials[0].node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (rng.random(), rng.random(), rng.random(), 1)
        
        dist = rng.uniform(0.5, 2.0)
        angle_x = rng.uniform(-0.25, 0.25)
        angle_y = rng.uniform(-0.25, 0.25)
        direction = camera.matrix_world.to_3x3() @ Vector((angle_x, angle_y, -1)).normalized()
        obj.location = camera.location + direction * dist
        scale = dist * rng.uniform(0.05, max_coverage)
        obj.scale = (scale, scale, scale)
        obj.rotation_euler = (rng.random() * math.tau, rng.random() * math.tau, rng.random() * math.tau)


def _apply_post_processing_noise(output_path, rng, prob, noise_types_str, int_min, int_max):
    if not noise_types_str or rng.random() > prob:
        return
    try:
        import numpy as np
        from PIL import Image
        import io
    except ImportError:
        print('Warning: numpy or PIL not installed, skipping noise.', file=sys.stderr)
        return

    noise_types = noise_types_str.split(',')
    chosen_type = rng.choice(noise_types)
    intensity = rng.uniform(int_min, int_max)

    with Image.open(output_path) as img:
        img_arr = np.array(img)
        
        if chosen_type == 'gaussian':
            noise = np.random.normal(0, intensity * 255, img_arr.shape).astype(np.float32)
            img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
            out_img = Image.fromarray(img_arr)
        elif chosen_type == 'salt_pepper':
            salt_prob = intensity / 2
            pepper_prob = intensity / 2
            rand = np.random.rand(*img_arr.shape[:2])
            if img_arr.shape[-1] == 4:
                # keep alpha channel unaffected if possible, or just apply to RGB
                rgb = img_arr[..., :3]
                rgb[rand < pepper_prob] = 0
                rgb[(rand >= pepper_prob) & (rand < pepper_prob + salt_prob)] = 255
                img_arr[..., :3] = rgb
            else:
                img_arr[rand < pepper_prob] = 0
                img_arr[(rand >= pepper_prob) & (rand < pepper_prob + salt_prob)] = 255
            out_img = Image.fromarray(img_arr)
        elif chosen_type == 'white':
            noise = np.random.uniform(-intensity * 255, intensity * 255, img_arr.shape).astype(np.float32)
            img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
            out_img = Image.fromarray(img_arr)
        elif chosen_type == 'jpeg':
            quality = int(100 - (intensity * 90))
            quality = max(1, min(100, quality))
            buffer = io.BytesIO()
            img.convert('RGB').save(buffer, format='JPEG', quality=quality)
            out_img = Image.open(buffer).convert('RGBA')
        else:
            out_img = img

        out_img.save(output_path)


def _render_sample(
    output_path,
    model_root,
    camera,
    lights,
    occluders,
    rng,
    args
):
    model_yaw = rng.uniform(-math.radians(8), math.radians(8))
    model_root.rotation_euler = (
        rng.uniform(-0.08, 0.08),
        rng.uniform(-0.08, 0.08),
        model_yaw,
    )

    radius = _sample_camera_radius(rng, args.camera_radius_min, args.camera_radius_max, args.close_view_ratio, args.far_view_ratio)
    theta = _sample_view_angle(rng, args.view_strategy, args.front_angle_degrees, args.fixed_view_angle_degrees) + model_yaw
    phi = rng.uniform(math.radians(64), math.radians(82))
    camera.location = (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )
    _look_at(camera, (0, 0, rng.uniform(-0.02, 0.18)))
    camera.data.lens = rng.uniform(36, 64)

    active_light_count = rng.randint(args.light_count_min, args.light_count_max)
    for idx, light in enumerate(lights):
        if idx < active_light_count:
            light.hide_render = False
            light.data.type = rng.choice(['POINT', 'SUN', 'SPOT', 'AREA'])
            light.location = (
                rng.uniform(-4.0, 4.0),
                rng.uniform(-4.0, 4.0),
                rng.uniform(2.0, 6.0),
            )
            light.data.energy = rng.uniform(100, 1500)
            light.data.color = (rng.random(), rng.random(), rng.random())
        else:
            light.hide_render = True

    _randomize_materials(rng)
    _set_world_background(rng, args.psychedelic_bg_prob)
    _randomize_occluders(rng, occluders, camera, args.occlusion_prob, args.occlusion_coverage_max)

    bpy.context.scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    
    _apply_post_processing_noise(
        output_path, rng, args.noise_prob, args.noise_types,
        args.noise_intensity_min, args.noise_intensity_max
    )


def main():
    args = _parse_args()
    rng = random.Random(args.seed)
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _load_model(model_path)
    model_root = _normalize_model()
    camera = _create_camera()
    lights = _create_lights(args.light_count_max)
    occluders = _create_occluders(args.occlusion_count_max)
    _setup_render(args.image_size, args.render_engine, args.device)

    for offset in range(args.samples):
        image_index = args.start_index + offset
        output_path = output_dir / f'{args.class_name}_{args.split}_{image_index:06d}.png'
        _render_sample(
            output_path,
            model_root,
            camera,
            lights,
            occluders,
            rng,
            args
        )


if __name__ == '__main__':
    main()
