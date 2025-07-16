bl_info = {
    "name": "Object Connector",
    "description": "Script with the ability to generate a connection between two objects in a scene. Supports multiple algorithms.",
    "author": "Matej Bujňák",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N-Panel) > Tool Tab",
    "category": "Object"
    }

import bpy
import bmesh
from bpy.props import IntProperty, StringProperty, CollectionProperty, EnumProperty, FloatProperty
from bpy.types import PropertyGroup
import mathutils
import os

# --- PROPERTY GROUP DEFINITION ---
# Defines a custom data structure that can be used in lists (CollectionProperty).
# A simple container for an integer, used to store face indices.
class FaceIndexPropertyGroup(PropertyGroup):
    """A custom property group to store a single integer."""
    index: IntProperty(name="Face Index")
    
    
def export_connector_as_obj(filepath, connector_obj):
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    
    # Ensure the connector object is active and selected
    bpy.context.view_layer.objects.active = connector_obj
    connector_obj.select_set(True)
    
    # Ensure the object is in Object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Export the selected object (only the connector) as OBJ
    bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True)
    connector_obj.select_set(False)


# --- LIVE VALIDATION LOGIC ---
# Pprovides immediate feedback to the user about their selections.
def validate_selection_util(scene, surface_index):
    """
    Analyzes a stored selection and returns its status ('VALID', 'INVALID', 'UNSET')
    and a human-readable message. This is the core validation logic.
    """
    # Determine which surface (1 or 2) to check based on the input.
    if surface_index == 1:
        obj_name, faces = scene.connector_obj_1, scene.connector_surfaces_1
    else:  # surface_index == 2
        obj_name, faces = scene.connector_obj_2, scene.connector_surfaces_2

    # First, check if a surface has been set at all.
    if not obj_name or not faces:
        return 'UNSET', "Not set"
        
    # Check if the object the user selected still exists in the scene.
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        return 'INVALID', f"Original object '{obj_name}' not found."

    def count_connected_components(target_obj, face_indices):
        bm = bmesh.new()
        bm.from_mesh(target_obj.data)
        bm.faces.ensure_lookup_table()
        faces_to_visit = set(face_indices)
        if not faces_to_visit:
            bm.free()
            return 0
        component_count = 0
        while faces_to_visit:
            component_count += 1
            # Pop an index
            face_index = faces_to_visit.pop()

            # Find the BMesh face that has .index == face_index
            start_face = None
            for f in bm.faces:
                if f.index == face_index:
                    start_face = f
                    break

            if start_face is None:
                # The face index doesn't exist anymore
                bm.free()
                return 0

            q = [start_face]
            while q:
                current_face = q.pop(0)
                for edge in current_face.edges:
                    for linked_face in edge.link_faces:
                        if linked_face.index in faces_to_visit:
                            faces_to_visit.remove(linked_face.index)
                            q.append(linked_face)
        bm.free()
        return component_count

    def get_loop_count(target_obj, face_indices):
        """
        Helper function to count the boundary loops of a specific set of faces.
        It does this in memory without creating temporary objects in the scene.
        """
        bm_orig = bmesh.new()
        bm_orig.from_mesh(target_obj.data)
        bm_orig.faces.ensure_lookup_table()

        # Get the BMesh face elements from the stored indices.
        faces_to_copy = []
        for i in face_indices:
            if i < len(bm_orig.faces):
                faces_to_copy.append(bm_orig.faces[i])
        
        if not faces_to_copy:
            bm_orig.free()
            return 0
        
        # Manually copy the geometry to a temporary bmesh to analyze it in isolation.
        bm_temp = bmesh.new()
        vmap = {}  # Maps original vertex indices to new vertices.
        
        # This loop rebuilds the selected faces in the temporary bmesh.
        # The 'vmap' dictionary is important to ensure that vertices shared
        # by multiple faces are only created once.
        for f in faces_to_copy:
            new_face_verts = []
            for v in f.verts:
                if v.index not in vmap:
                    vmap[v.index] = bm_temp.verts.new(v.co)
                new_face_verts.append(vmap[v.index])
            try:
                bm_temp.faces.new(new_face_verts)
            except ValueError:
                # This can happen if faces are degenerate; ignore them for validation.
                pass
        bm_orig.free()

        # Find the boundaries of the new temporary mesh by "crawling" the edges.
        # A boundary edge is one that is not manifold (not connected to exactly 2 faces).
        boundary_edges = set()
        for e in bm_temp.edges:
            if not e.is_manifold:
                boundary_edges.add(e)
                
        num_loops = 0
        # This loop continues until all boundary edges have been visited.
        while boundary_edges:
            # Each time this outer loop starts, we have found a new, separate boundary.
            num_loops += 1
            
            # Start a "crawl" from an arbitrary edge.
            path = [boundary_edges.pop()]
            while path:
                current_edge = path.pop()
                
                # Find all connected boundary edges and add them to the path to be processed.
                for v in current_edge.verts:
                    for linked_e in v.link_edges:
                        if linked_e in boundary_edges:
                            boundary_edges.remove(linked_e)
                            path.append(linked_e)
        bm_temp.free() # Clean up the temporary bmesh from memory.
        return num_loops

    # Create a set of face indices for faster lookups.
    face_indices = set()
    for f in faces:
        face_indices.add(f.index)
        
    topo_type = scene.connector_topology_type

    if topo_type == 'FACE_LOOP':
        num_components = count_connected_components(obj, face_indices)
        if num_components > 1:
            return 'INVALID', f"Face Loop must be one connected strip. Found {num_components} groups."
    
    # Add a specific check for Simple Bridge mode to enforce single-face selections.
    if topo_type == 'SIMPLE_BRIDGE':
        if len(faces) > 1:
            return 'INVALID', "Simple Bridge only supports a single face per surface."
    
    # Get the number of boundary loops from our selection.
    num_loops = get_loop_count(obj, face_indices)
    
    # Check if the number of found loops matches the requirement for the selected algorithm.
    if topo_type in ['SIMPLE_BRIDGE', 'CLOSED_CAP']:
        if num_loops == 1:
            return 'VALID', "1 boundary loop found."
        else:
            return 'INVALID', f"Expected 1 boundary loop, but found {num_loops}."
            
    elif topo_type == 'FACE_LOOP':
        if num_loops == 2:
            return 'VALID', "2 boundary loops found."
        else:
            return 'INVALID', f"Expected 2 boundary loops, but found {num_loops}."
        
    return 'INVALID', "Unknown topology error."

def run_validation(context):
    """A wrapper function that runs validation for both surfaces and compares them."""
    scene = context.scene
    
    # 1. Run individual validation for each surface first
    status1, msg1 = validate_selection_util(scene, 1)
    status2, msg2 = validate_selection_util(scene, 2)

    # 2. Run the comparison check, but only if both selections are individually valid
    if status1 == 'VALID' and status2 == 'VALID':
        obj1_name = scene.connector_obj_1
        obj2_name = scene.connector_obj_2

        if scene.connector_topology_type == 'SIMPLE_BRIDGE' and obj1_name == obj2_name:
            error_msg = "Simple Bridge cannot connect the same object."
            status1, msg1 = 'INVALID', error_msg
            status2, msg2 = 'INVALID', error_msg
        
        # The overlap check is only necessary if the selections are on the same object
        if obj1_name == obj2_name:
            faces1 = set()
            for f in context.scene.connector_surfaces_1:
                faces1.add(f.index)
            
            faces2 = set()
            for f in context.scene.connector_surfaces_2:
                faces2.add(f.index)

            if not faces1.isdisjoint(faces2):
                overlap_msg = "Surfaces cannot overlap."
                status1, msg1 = 'INVALID', overlap_msg
                status2, msg2 = 'INVALID', overlap_msg

    # 3. Store the final results in scene properties for the UI to read
    scene.connector_status_1 = status1
    scene.connector_error_1 = msg1
    scene.connector_status_2 = status2
    scene.connector_error_2 = msg2

def topology_type_update(self, context):
    """This function is an 'update callback' linked to the Topology Type dropdown."""
    run_validation(context)


# --- OPERATOR 1: SELECT AND STORE SURFACES ---
class OBJECT_OT_SelectConnectorSurface(bpy.types.Operator):
    """
    This operator runs when the user clicks "Set Surface 1" or "Set Surface 2".
    It identifies the selected faces on the active object in Edit Mode and saves their
    index numbers into a list in the scene for the main operator to use later.
    """
    # The bl_idname is the unique identifier for this operator in Blender.
    bl_idname = "object.select_connector_surface"
    bl_label = "Set Connector Surface"
    bl_options = {'REGISTER', 'UNDO'}

    # This property is set by the button in the UI (1 or 2) to know which list to save to.
    surface_index: IntProperty()

    def execute(self, context):
        # --- 1. Validation and Setup ---
        
        # Ensure the user is in Edit Mode on a mesh object.
        obj = context.object
        if not (obj and obj.mode == 'EDIT' and obj.type == 'MESH'):
            self.report({'ERROR'}, "Active object must be a mesh in Edit Mode.")
            return {'CANCELLED'}
        
        # Use BMesh to efficiently access mesh data.
        bm = bmesh.from_edit_mesh(obj.data)
        
        # Get a list of the currently selected faces.
        selected_faces = []
        for f in bm.faces:
            if f.select:
                selected_faces.append(f)
                
        if not selected_faces:
            self.report({'ERROR'}, "No faces selected.")
            return {'CANCELLED'}

        # --- 2. Store the Selection ---
        
        # Based on which button was pressed, store the object name and face indices
        # in the correct list.
        if self.surface_index == 1:
            # Clear any previously stored data for this surface.
            context.scene.connector_surfaces_1.clear()
            context.scene.connector_obj_1 = obj.name
            
            # Add the index of each selected face to the list.
            for face in selected_faces:
                context.scene.connector_surfaces_1.add().index = face.index
                
        elif self.surface_index == 2:
            # Clear any previously stored data for this surface.
            context.scene.connector_surfaces_2.clear()
            context.scene.connector_obj_2 = obj.name
            
            # Add the index of each selected face to the list.
            for face in selected_faces:
                context.scene.connector_surfaces_2.add().index = face.index
        
        # --- 3. Finalize ---
        
        # After storing the selection, immediately trigger the validation logic.
        run_validation(context)
        
        self.report({'INFO'}, f"Stored {len(selected_faces)} faces from {obj.name}.")
        return {'FINISHED'}


# --- HELPER FUNCTIONS ---
def get_boundary_loops_data(obj):
    """
    Analyzes an object's mesh to find all open boundaries (holes).
    Returns a list, where each item represents one boundary loop and contains
    its vertex coordinates in world-space and its total perimeter length.
    """
    if obj.type != 'MESH':
        return []
    
    # Create a temporary BMesh from the object's mesh data to analyze it.
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    
    # A boundary edge is an edge belonging to only one face.
    # We gather all of them into a set for efficient processing.
    boundary_edges = set()
    for e in bm.edges:
        if e.is_boundary:
            boundary_edges.add(e)
            
    all_loops_data = []

    # This loop continues until all boundary edges have been visited.
    while boundary_edges:
        # Start a "crawl" from an arbitrary edge.
        start_edge = boundary_edges.pop()
        current_edge_loop = [start_edge]
        
        # Trace the loop in one direction from the starting edge.
        v_curr = start_edge.verts[1]
        while True:
            # Find the next connected boundary edge.
            next_edge = next((e for e in v_curr.link_edges if e in boundary_edges), None)
            if not next_edge:
                break
            
            # Remove from set to avoid re-processing.
            boundary_edges.remove(next_edge)
            current_edge_loop.append(next_edge)
            v_curr = next_edge.other_vert(v_curr)
            
        # Trace in the other direction to complete the loop.
        v_curr = start_edge.verts[0]
        while True:
            next_edge = next((e for e in v_curr.link_edges if e in boundary_edges), None)
            if not next_edge:
                break

            boundary_edges.remove(next_edge)
            current_edge_loop.insert(0, next_edge)
            v_curr = next_edge.other_vert(v_curr)
            
        if not current_edge_loop:
            continue
        
        # Convert the ordered list of edges into an ordered list of vertices.
        vert_list = [current_edge_loop[0].verts[0]]
        last_vert = vert_list[0]
        for edge in current_edge_loop:
            next_vert = edge.other_vert(last_vert)
            vert_list.append(next_vert)
            last_vert = next_vert
        vert_list.pop() # Remove duplicated vertex at the end.
        
        # Convert vertex local coordinates to world coordinates for accurate calculations.
        coords = []
        for v in vert_list:
            coords.append(obj.matrix_world @ v.co)
        
        # Calculate the loop's perimeter to help with pairing later.
        size = 0
        for i in range(len(coords)):
            # Compare each vertex to the previous one in the list to get edge length.
            size += (coords[i] - coords[i-1]).length
            
        all_loops_data.append({"coords": coords, "size": size})
        
    bm.free() # Free the BMesh memory to prevent memory leaks.
    return all_loops_data

def resample_loop_coords(coords, target_count):
    """
    Takes a loop of coordinates and recalculates it to have a new number of evenly-spaced vertices.
    This is essential for connecting loops that have different vertex counts.
    """
    points = len(coords)
    if points < 2:
        return [coords[0]] * target_count if coords else []
    
    # 1. Parameterize the loop by measuring the cumulative distance at each original vertex.
    lengths = []
    for i in range(points):
        # The modulo operator (%) ensures the last vertex connects back to the first.
        lengths.append((coords[i] - coords[(i + 1) % points]).length)
        
    total_length = sum(lengths)
    if total_length == 0:
        return [coords[0]] * target_count # Handle degenerate case (zero-size loop).
        
    cumulative = [0.0]
    current_dist = 0
    for length in lengths:
        current_dist += length
        cumulative.append(current_dist)
    
    # 2. Create new points by stepping evenly along the total length.
    new_coords = []
    step = total_length / target_count
    for i in range(target_count):
        target_dist = i * step
        
        # Find which original segment the new point falls on.
        segment_idx = 0
        while segment_idx < points - 1 and target_dist >= cumulative[segment_idx + 1]:
            segment_idx += 1
            
        # Interpolate between the start and end points of that segment to find the new point's location.
        v1 = coords[segment_idx]
        v2 = coords[(segment_idx + 1) % points]
        
        dist1 = cumulative[segment_idx]
        dist2 = cumulative[segment_idx + 1]
        
        segment_len = dist2 - dist1
        local_t = (target_dist - dist1) / segment_len if segment_len > 0 else 0.0
        new_coords.append(v1.lerp(v2, local_t))
        
    return new_coords

def align_loops_by_cost(coords1, coords2):
    """
    Finds the best rotational alignment for two loops to prevent twisting.
    It tests every possible starting vertex and direction, choosing the one
    that results in the shortest connecting edges.
    """
    if not coords1 or not coords2 or len(coords1) != len(coords2):
        return None, None
        
    num_verts = len(coords1)
    best_offset = -1
    best_reversed = False
    min_cost = float('inf')
    
    reversed_coords2 = list(reversed(coords2))
    
    # Test every possible starting point on the second loop.
    for offset in range(num_verts):
        # Calculate the "cost" (sum of squared distances) for the loop as-is.
        cost = 0
        for i in range(num_verts):
            v1 = coords1[i]
            # The offset rotates the starting point of the second loop.
            v2 = coords2[(i + offset) % num_verts]
            cost += (v1 - v2).length_squared
            
        if cost < min_cost:
            min_cost = cost
            best_offset = offset
            best_reversed = False
        
        # Calculate the cost for the reversed loop.
        cost = 0
        for i in range(num_verts):
            v1 = coords1[i]
            v2 = reversed_coords2[(i + offset) % num_verts]
            cost += (v1 - v2).length_squared
            
        if cost < min_cost:
            min_cost = cost
            best_offset = offset
            best_reversed = True
            
    # Construct and return the best aligned loop configuration.
    final_coords2 = reversed_coords2 if best_reversed else coords2
    return coords1, final_coords2[best_offset:] + final_coords2[:best_offset]

class OBJECT_OT_GenerateConnector(bpy.types.Operator):
    """
    The main operator that holds the logic for all three connection algorithms.
    It runs when the user clicks the "Generate Connector" button.
    """
    bl_idname = "object.generate_loop_connection"
    bl_label = "Generate Connector"
    bl_options = {'REGISTER', 'UNDO'}

    simple_bridge_cuts: IntProperty(
        name="Number of Cuts",
        description="Number of intermediate edge loops",
        default=0,
        min=0
    )
    simple_bridge_interpolation: EnumProperty(
        name="Interpolation",
        description="Algorithm used for interpolation",
        items=[
            ('LINEAR', "Linear", "Simple linear interpolation"),
            ('PATH', "Blend Path", "Path-based blending")
        ],
        default='LINEAR'
    )
    simple_bridge_smoothness: FloatProperty(
        name="Smoothness",
        description="Smoothness of the Blend Surface interpolation",
        default=1.0,
        min=0.0,
        max=5.0
    )
    simple_bridge_profile_shape: EnumProperty(
        name="Profile Shape",
        description="Shape of the profile",
        items=[
            ('LINEAR', "Linear", "Linear profile"),
            ('SHARP', "Sharp", "Sharp profile"),
            ('INVERSE_SQUARE', "Inverse Square", "Inverse square profile"),
            ('SPHERE', "Sphere", "Spherical profile"),
            ('ROOT', "Root", "Square root profile"),
            ('SMOOTH', "Smooth", "Smooth profile"),
        ],
        default='LINEAR'
    )
    simple_bridge_profile_factor: FloatProperty(
        name="Profile Factor",
        description="How much the profile shape influences the bridge",
        default=0.0,
        min=-5.0,
        max=5.0
    )
    
    def draw(self, context):
        """Method to draw Redo Last Operation panel for Simple bridge."""
        layout = self.layout
        scene = context.scene

        # Display settings if Simple Bridge was selected
        if scene.connector_topology_type == 'SIMPLE_BRIDGE':
            layout.label(text="Simple Bridge Options:")
            
            # Drawing properties from "self", not from "scene"
            layout.prop(self, "simple_bridge_cuts")
            layout.prop(self, "simple_bridge_interpolation")

            if self.simple_bridge_interpolation in ['BLEND_SURFACE', 'BLEND_PATH']:
                layout.prop(self, "simple_bridge_smoothness")

            layout.prop(self, "simple_bridge_profile_shape")
            if self.simple_bridge_profile_shape != 'FLAT':
                layout.prop(self, "simple_bridge_profile_factor")

    def duplicate_face_loop(self, context, obj, face_indices, part_name):
        """
        Helper function to duplicate a set of faces from an object
        and separate them into a new, independent object.
        """
        # Ensure we are in Object Mode before switching objects.
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
            
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')

        # Select the specific faces using the stored indices.
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        for f in bm.faces:
            f.select = f.index in face_indices
        bmesh.update_edit_mesh(obj.data)

        # Keep track of objects in the scene before duplicating.
        objects_before = set(context.scene.objects)
        
        # Duplicate the selected faces and separate them.
        bpy.ops.mesh.duplicate()
        bpy.ops.mesh.separate(type='SELECTED')
        bpy.ops.object.mode_set(mode='OBJECT')

        # Find the newly created object by comparing the scene contents.
        objects_after = set(context.scene.objects)
        new_obj = (objects_after - objects_before).pop()
        
        new_obj.name = f"{obj.name}_{part_name}"
        
        # Select and make the new object active for clarity.
        bpy.ops.object.select_all(action='DESELECT')
        new_obj.select_set(True)
        context.view_layer.objects.active = new_obj
        return new_obj

    def execute_simple_bridge(self, context, end_cap_1, end_cap_2):
        """Handles the 'Simple Bridge' algorithm."""
        # Join the two end caps into a single object.
        bpy.ops.object.select_all(action='DESELECT')
        end_cap_1.select_set(True)
        end_cap_2.select_set(True)
        context.view_layer.objects.active = end_cap_1
        bpy.ops.object.join()
        
        final_obj = context.active_object
        final_obj.name = "Simple_Connector"
        context.scene.connector_obj = final_obj.name

        # Use Blender's robust Bridge Edge Loops tool to generate the connection.
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_mode(type='EDGE')
        bpy.ops.mesh.select_non_manifold()

        # Reading properties from self
        bpy.ops.mesh.bridge_edge_loops(
            number_cuts=self.simple_bridge_cuts, 
            interpolation=self.simple_bridge_interpolation,
            smoothness=self.simple_bridge_smoothness,
            profile_shape=self.simple_bridge_profile_shape,
            profile_shape_factor=self.simple_bridge_profile_factor
        )

        # Weld seams and recalculate normals
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        return final_obj
    
    def execute_advanced_loft(self, context, end_cap_1, end_cap_2):
        """Handles the 'Advanced Loft' algorithms (Face Loop and Closed Cap)."""
        # Get the boundary data from the temporary end caps.
        loops1_data = get_boundary_loops_data(end_cap_1)
        loops2_data = get_boundary_loops_data(end_cap_2)
        
        if not loops1_data or not loops2_data:
            raise Exception("Could not find boundary loops on duplicated parts.")

        # Pair the loops from each surface based on the chosen algorithm.
        topology_type = context.scene.connector_topology_type
        unprocessed_pairs = []
        
        if topology_type == 'FACE_LOOP':
            loops1_data.sort(key=lambda d: d['size'])
            loops2_data.sort(key=lambda d: d['size'])
            if len(loops1_data) != len(loops2_data):
                self.report({'WARNING'}, f"Mismatched boundary count.")
            
            # Pair smallest loop with smallest, largest with largest, etc.
            limit = min(len(loops1_data), len(loops2_data))
            for i in range(limit):
                unprocessed_pairs.append((loops1_data[i]["coords"], loops2_data[i]["coords"]))
        elif topology_type == 'CLOSED_CAP':
            if len(loops1_data) != 1 or len(loops2_data) != 1:
                raise Exception(f"For 'Closed Cap' mode, each selection must have one boundary. Found {len(loops1_data)} and {len(loops2_data)}.")
            unprocessed_pairs.append((loops1_data[0]["coords"], loops2_data[0]["coords"]))
        
        if not unprocessed_pairs:
            raise Exception("No valid loop pairs found.")

        # Build the bridge geometry in a new, clean BMesh.
        bm = bmesh.new()
        num_segments = context.scene.connector_number_segments
        num_cuts = context.scene.connector_number_cuts
        
        for coords1, coords2 in unprocessed_pairs:
            # Resample, Align, and Interpolate to get the final vertex positions.
            if len(coords1) != len(coords2):
                max_count = max(len(coords1), len(coords2))
                coords1 = resample_loop_coords(coords1, max_count)
                coords2 = resample_loop_coords(coords2, max_count)
            v1_coords, v2_coords = align_loops_by_cost(coords1, coords2)
            
            all_profiles = []
            total_steps = (num_segments + 1) * (num_cuts + 1)
            
            # Create all the vertices for the bridge profiles.
            for step in range(total_steps + 1):
                linear_t = step / total_steps if total_steps > 0 else 0.0
                final_t = linear_t # Possibility to expand towards other interpolation functions.
                
                profile_coords = []
                for i in range(len(v1_coords)):
                    profile_coords.append(v1_coords[i].lerp(v2_coords[i], final_t))
                
                profile_verts = []
                for co in profile_coords:
                    profile_verts.append(bm.verts.new(co))
                all_profiles.append(profile_verts)
                
            # Create faces between the generated profiles.
            for p_idx in range(len(all_profiles) - 1):
                loop_a = all_profiles[p_idx]
                loop_b = all_profiles[p_idx+1]
                for v_idx in range(len(loop_a)):
                    v1 = loop_a[v_idx]
                    v2 = loop_a[(v_idx + 1) % len(loop_a)]
                    v3 = loop_b[(v_idx + 1) % len(loop_b)]
                    v4 = loop_b[v_idx]
                    try:
                        bm.faces.new((v1, v2, v3, v4))
                    except ValueError:
                        print(f"Warning: Could not create face.")

        # Create a temporary bridge object from the BMesh.
        bridge_mesh = bpy.data.meshes.new("Connector_Bridge_Mesh")
        bm.to_mesh(bridge_mesh)
        bm.free()
        bridge_obj = bpy.data.objects.new("Connector_Bridge", bridge_mesh)
        context.collection.objects.link(bridge_obj)
        
        # Join the end caps and the bridge into one object.
        bpy.ops.object.select_all(action='DESELECT')
        end_cap_1.select_set(True)
        end_cap_2.select_set(True)
        bridge_obj.select_set(True)
        context.view_layer.objects.active = bridge_obj
        bpy.ops.object.join()
        
        final_obj = context.active_object
        final_obj.name = "Advanced_Connector"
        bpy.context.scene.connector_obj = final_obj.name
    
        # Weld the seams to create a single, continuous mesh.
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.mesh.flip_normals()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        return final_obj
        
    def execute(self, context):
        # Ensure we start in Object Mode to prevent context errors.
        if context.active_object and context.active_object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # --- 1. Pre-generation Validation ---
        # First, check the status set by the live validation system.
        # This implicitly handles the overlap case because the status will be 'INVALID'
        status1 = context.scene.connector_status_1
        status2 = context.scene.connector_status_2
        if status1 != 'VALID' or status2 != 'VALID':
            self.report({'ERROR'}, f"Invalid selection(s). Please check UI for errors.")
            return {'CANCELLED'}

        # Get the original objects and face selections from the scene.
        orig_obj1_name = context.scene.connector_obj_1
        orig_obj2_name = context.scene.connector_obj_2
        orig_obj1 = bpy.data.objects.get(orig_obj1_name)
        orig_obj2 = bpy.data.objects.get(orig_obj2_name)
        
        faces1 = set()
        for f in context.scene.connector_surfaces_1:
            faces1.add(f.index)

        faces2 = set()
        for f in context.scene.connector_surfaces_2:
            faces2.add(f.index)
        
        topology_type = context.scene.connector_topology_type
        final_obj = None
        
        # A 'try...finally' block ensures that the cleanup code at the end
        # will run even if an error occurs during the operation.
        try:
            # --- 2. Create End Caps ---
            # All modes start by creating temporary end caps from the selected faces.
            # This ensures the original objects are never modified.
            end_cap_1 = self.duplicate_face_loop(context, orig_obj1, faces1, "EndCap1")
            end_cap_2 = self.duplicate_face_loop(context, orig_obj2, faces2, "EndCap2")
            
            # --- 3. Execute Selected Algorithm ---
            # Branch to the correct algorithm based on the user's choice.
            if topology_type == 'SIMPLE_BRIDGE':
                final_obj = self.execute_simple_bridge(context, end_cap_1, end_cap_2)
            else: 
                final_obj = self.execute_advanced_loft(context, end_cap_1, end_cap_2)
            
            if final_obj:
                context.scene.connector_obj = final_obj.name

            self.report({'INFO'}, "Connector generation successful.")
            
        except Exception as e:
            self.report({'ERROR'}, f"Operation failed: {e}")
            # Clean up temporary objects if an error occurs.
            if 'end_cap_1' in locals() and end_cap_1.name in bpy.data.objects:
                bpy.data.objects.remove(end_cap_1)
            if 'end_cap_2' in locals() and end_cap_2.name in bpy.data.objects:
                bpy.data.objects.remove(end_cap_2)
            return {'CANCELLED'}
            
        finally:
            # --- 4. FINALIZE AND SELECT NEW OBJECT ---
            # Data is not cleared, but selection is not restored.
            # Instead, we select the new connector.
            
            if context.object and context.object.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            
            # Select only the final generated object.
            bpy.ops.object.select_all(action='DESELECT')
            if final_obj and final_obj.name in bpy.data.objects:
                final_obj.select_set(True)
                context.view_layer.objects.active = final_obj
                bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')

        return {'FINISHED'}
    
class OBJECT_OT_ExportConnector(bpy.types.Operator):
    """
    Exports the currently generated connector to a file.
    
    Uses the name, path, and format specified in the UI panel.
    An object must be generated first before it can be exported.
    """
    bl_idname = "object.export_connector"
    bl_label = "Export Connector"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            scene = context.scene
            connector_obj_name = scene.connector_obj
            connector_obj = bpy.data.objects.get(connector_obj_name)

            if connector_obj is None:
                self.report({'ERROR'}, "No connector object found to export.")
                return {'CANCELLED'}

            # Get the user-defined path and format
            export_path = bpy.path.abspath(scene.export_path)
            export_format = scene.export_format
            file_extension = export_format.lower()

            # Ensure the directory exists
            if not os.path.isdir(export_path):
                self.report({'ERROR'}, "The specified export directory does not exist.")
                return {'CANCELLED'}
            
            # Use the custom export name if provided, otherwise use the object's name
            export_name = scene.connector_export_name.strip()
            if not export_name:
                export_name = connector_obj.name # Fallback to the object's name

            # Construct the full filepath using the correct 'export_name' variable
            filename = f"{export_name}.{file_extension}"
            filepath = os.path.join(export_path, filename)

            # Deselect all and select only the connector
            bpy.ops.object.select_all(action='DESELECT')
            connector_obj.select_set(True)
            context.view_layer.objects.active = connector_obj

            # Call the appropriate exporter
            if export_format == 'OBJ':
                bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True)
            elif export_format == 'STL':
                bpy.ops.export_mesh.stl(filepath=filepath, use_selection=True)

            connector_obj.select_set(False)

            self.report({'INFO'}, f"Connector exported to {filepath}")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {str(e)}")
            return {'CANCELLED'}
        
# --- UI PANEL ---
class OBJECT_PT_ObjectConnectorPanel(bpy.types.Panel):
    """Creates the addon's UI panel in the 3D View's sidebar."""
    
    # The bl_idname must be unique to this panel.
    # The bl_label is the title of the panel shown in the UI.
    # The bl_category is the tab name in the sidebar (the "N-Panel").
    # The bl_space_type and bl_region_type define where the panel can appear.
    bl_idname = "OBJECT_PT_ObjectConnectorPanel"
    bl_label = "Object Connector"
    bl_category = "Tool"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    def draw(self, context):
        """This method is called by Blender to draw the panel's contents."""
        layout, scene = self.layout, context.scene
        
        # Use a single main column for a clean, sequential layout.
        col = layout.column(align=True)
        
        # --- Section 1: Inputs ---
        col.label(text="1. Set Surfaces to Connect:")
        row = col.row(align=True)
        row.operator("object.select_connector_surface", text="Set Surface 1").surface_index = 1
        row.operator("object.select_connector_surface", text="Set Surface 2").surface_index = 2
        
        # --- Draw Object 1 Property with Status Indication ---
        row = col.row(align=True)
        status1 = scene.connector_status_1
        is_valid = (status1 == 'VALID')
        
        # The 'alert' property colors the row red if the selection is not valid.
        row.alert = not is_valid
        
        # Display an icon based on the validation status.
        row.label(text="", icon='CHECKMARK' if is_valid else 'ERROR' if status1 == 'INVALID' else 'CANCEL')
        sub = row.row()
        sub.enabled = False # Make the text field read-only.
        sub.prop(scene, "connector_obj_1", text="Obj 1")

        # --- Draw Object 2 Property with Status Indication ---
        row = col.row(align=True)
        status2 = scene.connector_status_2
        is_valid = (status2 == 'VALID')
        row.alert = not is_valid
        row.label(text="", icon='CHECKMARK' if is_valid else 'ERROR' if status2 == 'INVALID' else 'CANCEL')
        sub = row.row()
        sub.enabled = False
        sub.prop(scene, "connector_obj_2", text="Obj 2")
        
        # --- Dedicated Error Box ---
        # If either selection is invalid, create a red error box
        # and display the specific error message(s).
        if scene.connector_status_1 == 'INVALID' or scene.connector_status_2 == 'INVALID':
            box = layout.box()
            box.alert = True
            if scene.connector_status_1 == 'INVALID':
                box.label(text=f"Surface 1: {scene.connector_error_1}", icon='ERROR')
            if scene.connector_status_2 == 'INVALID':
                box.label(text=f"Surface 2: {scene.connector_error_2}", icon='ERROR')

        col.separator()
        
        # --- Section 2: Algorithm and Parameters ---
        col.label(text="2. Choose Algorithm & Settings:")
        col.prop(scene, "connector_topology_type", text="")
        
        # Determine if the advanced options should be interactive.
        is_advanced_mode = (scene.connector_topology_type != 'SIMPLE_BRIDGE')
        
        # A box for the advanced settings, which can be greyed out.
        box = col.box()
    
        #The 'active' property disables the UI elements inside the box.
        box.active = is_advanced_mode
        
        sub_col = box.column()
        sub_col.label(text="Advanced Options:")
        sub_col.prop(scene, "connector_number_segments", text="Intermediate Segments")
        sub_col.prop(scene, "connector_number_cuts", text="Cuts per Segment")

        col.separator()
        
        # --- Section 3: Execution ---
        col.label(text="3. Generate:")
        col.operator("object.generate_loop_connection", text="Generate Connector", icon='PLAY')
        col.separator()
        
        # --- Section 4: Export ---
        col.label(text="4. Export Connector:")
        col.prop(scene, "connector_export_name", text="Export Name")
        col.prop(scene, "export_path", text="Export Path")
        
        col.prop(scene, "export_format", text="Export Format")
        col.operator("object.export_connector", text="Export Connector", icon='EXPORT')

# --- REGISTRATION ---
# A tuple containing all the classes that need to be registered with Blender.
# This makes the register() and unregister() functions cleaner.
classes = (
    FaceIndexPropertyGroup,
    OBJECT_OT_SelectConnectorSurface,
    OBJECT_OT_GenerateConnector,
    OBJECT_OT_ExportConnector,
    OBJECT_PT_ObjectConnectorPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
        
    bpy.types.Scene.connector_surfaces_1 = CollectionProperty(type=FaceIndexPropertyGroup)
    bpy.types.Scene.connector_surfaces_2 = CollectionProperty(type=FaceIndexPropertyGroup)
    bpy.types.Scene.connector_obj_1 = StringProperty(name="Source Object 1")
    bpy.types.Scene.connector_obj_2 = StringProperty(name="Source Object 2")
    bpy.types.Scene.connector_status_1 = StringProperty(name="Status 1", default="UNSET")
    bpy.types.Scene.connector_status_2 = StringProperty(name="Status 2", default="UNSET")
    bpy.types.Scene.connector_error_1 = StringProperty(name="Error 1")
    bpy.types.Scene.connector_error_2 = StringProperty(name="Error 2")
    bpy.types.Scene.connector_number_segments = IntProperty(
        name="Intermediate Segments", 
        default=1, 
        min=0, 
        max=100,
        description="Number of divisions along the length of the connector"
    )
    bpy.types.Scene.connector_number_cuts = IntProperty(
        name="Cuts per Segment", 
        default=3, 
        min=0, 
        max=100,
        description="Number of divisions along the length of each segment"
    )
    bpy.types.Scene.connector_topology_type = EnumProperty(
        name="Topology Type",
        description="The type of algorithm to use for the connection",
        items=[
            ('FACE_LOOP', "Face Loop", "For connecting rings of faces (2 boundaries)."),
            ('CLOSED_CAP', "Closed Cap", "For simple surfaces (1 boundary). Uses biased interpolation."),
            ('SIMPLE_BRIDGE', "Simple Bridge", "Robustly connects simple surfaces (1 boundary) using Blender's tool."),
        ],
        default='FACE_LOOP',
        update=topology_type_update
    )
    bpy.types.Scene.connector_obj = bpy.props.StringProperty(name="Connector", default="")
    bpy.types.Scene.export_format = EnumProperty(
        name="Format",
        description="Choose the export file format",
        items=[
            ('OBJ', "OBJ", "Export as a .obj file"),
            ('STL', "STL", "Export as a .stl file"),
        ],
        default='OBJ'
    )
    
    bpy.types.Scene.export_path = StringProperty(
        name="Export Path",
        description="Directory to save the exported file",
        default="//",  # Defaults to the same directory as the .blend file
        subtype='DIR_PATH'
    )
    
    bpy.types.Scene.connector_export_name = StringProperty(
        name="Connector Name",
        description="Set the name for the generated connector object and file",
        default="Connector"
    )
    bpy.app.handlers.depsgraph_update_post.append(topology_type_update)

def unregister():
    for handler in bpy.app.handlers.depsgraph_update_post:
        if handler.__name__ == 'topology_type_update':
            bpy.app.handlers.depsgraph_update_post.remove(handler)
            break
            
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
        
    del bpy.types.Scene.connector_surfaces_1
    del bpy.types.Scene.connector_surfaces_2
    del bpy.types.Scene.connector_obj_1
    del bpy.types.Scene.connector_obj_2
    del bpy.types.Scene.connector_status_1
    del bpy.types.Scene.connector_status_2
    del bpy.types.Scene.connector_error_1
    del bpy.types.Scene.connector_error_2
    del bpy.types.Scene.connector_number_segments
    del bpy.types.Scene.connector_number_cuts
    del bpy.types.Scene.connector_topology_type
    del bpy.types.Scene.connector_obj
    del bpy.types.Scene.export_format
    del bpy.types.Scene.export_path
    del bpy.types.Scene.connector_export_name

if __name__ == "__main__":
    register()