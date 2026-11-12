import blenderproc as bproc  # must be the very first line, BlenderProc checks this

# Print Blender's colour-management settings as BlenderProc leaves them after init.
import bpy

bproc.init()
vs = bpy.context.scene.view_settings
print(f"view_transform = {vs.view_transform}, look = {vs.look}, exposure = {vs.exposure}, gamma = {vs.gamma}")
print(f"display_device = {bpy.context.scene.display_settings.display_device}")