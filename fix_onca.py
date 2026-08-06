import bpy
import sys

try:
    filepath = "assets/blender_models/onca/onca.blend"
    bpy.ops.wm.open_mainfile(filepath=filepath)
    print("Opened file")
    
    old_scene = bpy.context.window.scene
    new_scene = bpy.data.scenes.new(name="FixedScene")
    
    for obj in old_scene.objects:
        new_scene.collection.objects.link(obj)
        
    bpy.context.window.scene = new_scene
    bpy.data.scenes.remove(old_scene)
    
    new_scene.render.image_settings.file_format = 'PNG'
    
    bpy.ops.wm.save_as_mainfile(filepath=filepath)
    print("Saved file successfully.")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
