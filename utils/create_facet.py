import numpy as np
import trimesh

def create_parametric_facet_mesh(rho, phi, h, width=0.30):
    """
    Crea un facet rectangular vertical.

    Parámetros:
    rho   : distancia radial desde el borde/esquina [m]
    phi   : ángulo polar [rad]
    h     : altura del facet desde el suelo [m]
    width : ancho horizontal del facet [m]

    El facet va desde z=0 hasta z=h.
    """

    # Centro horizontal del facet
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    # Dirección radial desde el origen hacia el facet
    radial = np.array([np.cos(phi), np.sin(phi), 0.0])

    # Dirección tangencial, perpendicular a la radial
    tangent = np.array([-np.sin(phi), np.cos(phi), 0.0])

    # Centro de la base del facet
    base_center = np.array([x, y, 0.0])

    # Cuatro vértices del rectángulo
    v0 = base_center - 0.5 * width * tangent
    v1 = base_center + 0.5 * width * tangent
    v2 = v1 + np.array([0.0, 0.0, h])
    v3 = v0 + np.array([0.0, 0.0, h])

    vertices = np.vstack([v0, v1, v2, v3])

    # Caras triangulares.
    # Este orden hace que la normal apunte aproximadamente hacia el origen,
    # es decir, hacia la región visible/láser.
    faces = np.array([
        [0, 2, 1],
        [0, 3, 2]
    ])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    return mesh