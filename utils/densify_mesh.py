def densify_mesh_if_needed(mesh, min_triangles=10000):
    while len(mesh.faces) < min_triangles:
        mesh = mesh.subdivide()
    return mesh