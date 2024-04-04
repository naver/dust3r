import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

def point_inside_cone(point, position, direction, fov, height):
    """
    Check if a point is inside a cone defined by its apex position, direction, fov, and height.
    """
    point = np.array(point)
    position = np.array(position)
    direction = np.array(direction) / np.linalg.norm(direction)  # Normalize direction

    # Vector from apex to point
    apex_to_point = point - position

    # Distance of the point from the apex along the cone's direction
    distance_along_direction = np.dot(apex_to_point, direction)

    # Check if the point is within the cone's height and in front of the apex
    if distance_along_direction < 0 or distance_along_direction > height:
        return False

    # Calculate the maximum allowed distance from the cone's axis for the point's height
    fov_rad = np.radians(fov)
    max_radius_at_point_height = (distance_along_direction / height) * (height * np.tan(fov_rad / 2))

    # Project apex_to_point onto a plane perpendicular to the direction to find its distance from the axis
    perpendicular_component = apex_to_point - distance_along_direction * direction
    distance_from_axis = np.linalg.norm(perpendicular_component)

    # Check if the point is within the allowed radius at its height
    return distance_from_axis <= max_radius_at_point_height

def generate_random_point_in_cone(position, direction, fov, height):
    """
    Generate a random point inside a cone.

    Parameters:
    - position: The position vector of the cone's apex.
    - direction: The cone's direction vector.
    - fov: The field of view of the cone in degrees.
    - height: The height of the cone from the apex.

    Returns:
    - A point within the cone as a numpy array.
    """
    fov_rad = np.radians(fov - 0.00001)
    
    # Calculate the radius of the base of the cone
    base_radius = height * np.tan(fov_rad / 2)
    
    # Generate a random height within the cone
    z = np.random.uniform(0, height)
    
    # Calculate the radius at this height
    radius_at_z = (z / height) * base_radius
    
    # Generate a random angle theta
    theta = np.random.uniform(0, 2 * np.pi)
    
    # Generate a random radius within the maximum radius at height z
    random_radius = np.random.uniform(0, radius_at_z)
    
    # Calculate x and y coordinates based on the random radius and angle
    x = random_radius * np.cos(theta)
    y = random_radius * np.sin(theta)
    
    # Create the point in the cone's local space
    point_local = np.array([x, y, z])
    
    # Normalize direction vector
    direction_norm = np.array(direction) / np.linalg.norm(direction)
    z_axis = np.array([0, 0, 1])
    if not np.allclose(direction_norm, z_axis):
        # Calculate rotation required from Z-axis to direction vector
        axis = np.cross(z_axis, direction_norm)
        angle = np.arccos(np.clip(np.dot(z_axis, direction_norm), -1.0, 1.0)) # Use np.clip to ensure the dot product is within valid range
        rotation = Rotation.from_rotvec(axis * angle)
        point_local = rotation.apply(point_local)
    
    # Translate the point to the cone's position in space
    point_global = point_local + np.array(position)
    
    return point_global

def cone_IoU(cone1, cone2, num_samples=10000):
    """
    Approximate the volume overlap of two cones using Monte Carlo simulation.
    """
    total_count = num_samples * 2
    overlap_count = 0
    for _ in range(num_samples):
        if (point_inside_cone(generate_random_point_in_cone(**cone1), **cone2)):
            overlap_count += 1
        if (point_inside_cone(generate_random_point_in_cone(**cone2), **cone1)):
            overlap_count += 1
    return (overlap_count / total_count)

def get_camera_visual_cone(transform_matrix, intrinsics_matrix, height):
    intrinsics_matrix = np.array(intrinsics_matrix)
    transform_matrix = np.array(transform_matrix)
    fx = intrinsics_matrix[0, 0]  # Focal length in x
    fy = intrinsics_matrix[1, 1]  # Focal length in y
    cx = intrinsics_matrix[0, 2]  # Principal point x
    cy = intrinsics_matrix[1, 2]  # Principal point y
    fov_x = 2 * np.arctan(cx / fx)
    fov_y = 2 * np.arctan(cy / fy)
    fov = min(fov_x, fov_y)
    position = transform_matrix[:3, 3]
    direction = transform_matrix[:3, 2]  
    return {
        "position": position,
        "direction": direction,
        "fov": np.degrees(fov),
        "height": height
    }

def create_cone_vis(position, direction, fov, height=1.0):
    """
    Create a cone mesh given a ray (position and direction), FOV, and height.

    Parameters:
    - position: The position vector of the ray's origin (apex of the cone).
    - direction: The directional vector of the ray.
    - fov: The field of view in degrees.
    - height: The height of the cone from the apex.

    Returns:
    - A trimesh object representing the cone.
    """
    # Convert FOV from degrees to radians and calculate base radius
    fov_rad = np.radians(fov)
    radius = height * np.tan(fov_rad / 2)
    # Create the cone mesh
    cone_mesh = trimesh.creation.cone(radius=radius, height=height)
    # Align cone with direction vector
    # Calculate rotation required from the Z-axis to the direction vector
    direction = np.array(direction) / np.linalg.norm(direction)  # Normalize direction
    z_axis = np.array([0, 0, 1])
    if np.allclose(direction, z_axis):
        # Direction is already aligned with the Z-axis, no rotation needed
        rotation_matrix = np.eye(4)
    else:
        axis = np.cross(z_axis, direction)
        angle = np.arccos(np.dot(z_axis, direction))
        rotation = Rotation.from_rotvec(axis * angle)
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = rotation.as_matrix()
    # Apply rotation
    cone_mesh.apply_transform(rotation_matrix)
    # Translate cone to position
    cone_mesh.apply_translation(position)

    return cone_mesh

def compute_camera_iou(
    transform_matrix1,
    transform_matrix2,
    intrinsics_matrix1,
    intrinsics_matrix2,
    height=100,
    num_samples=10000,
    should_vis=False,
):
    cone1 = get_camera_visual_cone(transform_matrix1, intrinsics_matrix1, height=height)
    cone2 = get_camera_visual_cone(transform_matrix2, intrinsics_matrix2, height=height)
    cone_mesh = None
    if should_vis:
        cone1_mesh = create_cone_vis(**cone1)
        cone2_mesh = create_cone_vis(**cone2)
        cone_mesh = (cone1_mesh+cone2_mesh)
    return cone_IoU(cone1, cone2, num_samples=num_samples), cone_mesh


if __name__ == "__main__":
    transform_matrix1 = [[ 9.9760526e-01, -4.5130432e-02,  5.2411411e-02,  7.5428978e+01],
            [ 5.8392942e-02,  1.4345439e-01, -9.8793274e-01, -1.3015556e+02],
            [ 3.7067186e-02,  9.8862737e-01,  1.4574616e-01,  4.0148785e+01],
            [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]

    transform_matrix2 = [[ 9.7472835e-01, -1.6498064e-01,  1.5061878e-01,  5.9490425e+01],
            [ 1.5712430e-01,  2.7041817e-02, -9.8720855e-01, -1.5416634e+02],
            [ 1.5879729e-01,  9.8592603e-01,  5.2280892e-02,  3.6160526e+01],
            [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]

    intrinsics_matrix = [[269.04523,    0.,       111.65169 ],
                        [  0.,       201.78392,  112.151695],
                        [  0.,         0.,         1.      ]]

    import time

    n_iters = 1000
    runtimes = []
    ious = []
    for i in range(n_iters):
        start = time.time()
        iou = compute_camera_iou(transform_matrix1, transform_matrix2, intrinsics_matrix, intrinsics_matrix, height1=100, height2=100, num_samples=100)
        end = time.time()
        runtimes.append((end - start) * 1000)
        ious.append(iou)
    print(f"got IoU of {iou:.4f} in average {np.mean(runtimes):.0f} ms ({np.std(ious):.3f} std)")
