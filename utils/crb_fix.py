import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix
from utils.noise import add_sensor_noise, add_background_noise, add_poisson_noise
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.check_overlaps import check_overlaps
from utils.create_wall import create_wall_mesh
from utils.get_files import get_obj_files
from utils.sparse_wall import create_sparse_wall
from utils.densify_mesh import densify_mesh_if_needed


# ## ⚠️⚠️ Parametros no modificables ⚠️⚠️
# 



c = 299792458

object_folder = 'objects'

ymin = 0
zmin = 0


# ## ✅✅ Parámetros de entrada (modificables) ✅✅


xmin = -1.5
xmax = 1.5
ymax = 3
zmax = 3

camera_FOV = 0.5
cam_pixel_dim = 64
bin_size = 3.9e-10
laser_intensity = 1000
hide_walls = True
SNR_dB = 30
SBR = 5
poisson_scale_factor = 1000
add_noise = False

MESH_MIN_TRIANGLES = 5000

# Posiciones de objetos (debes tener los .obj en la carpeta 'objects/')
object_positions = [
    {
        'obj_file': 'facet.obj',
        'xcoord': 0.6,
        'ycoord': 2,
        'zcoord': 0,
        'w': 0.5,
        'yaw': 0,
        'pitch': 1.57,
        'roll': 0,
    }
]

def simulation(xmin, xmax, ymax, zmax, camera_FOV, cam_pixel_dim, bin_size, laser_intensity, object_positions, hide_walls, SNR_dB, SBR, poisson_scale_factor, add_noise, uploaded_objs=None, object_meshes=None):
    objects = []  # Inicializa la lista que contendrá todos los objetos de la escena.
    scene_objects = []  # Inicializa la lista que contendrá solo los objetos principales (modelos 3D).

    # ----- Configuración de la cámara SPAD -----
    camera_FOV_center = [0, -camera_FOV / 2, 0]  # Define el centro del campo de visión (FOV) en coordenadas 3D.
    FOV_radius = camera_FOV / cam_pixel_dim  # Calcula el "radio" o tamaño de cada píxel dentro del FOV.

    # ----- Configuración del láser -----
    laser_pos = np.array([0, 0, 0])  # Define la posición del láser en el origen.
    laser_normal = np.array([0, 0, 1])  # Define la dirección en la que apunta el láser (eje Z positivo).

    # ----- Configuración de la simulación -----
    wall_discr = c / 2 * bin_size / 4  # Calcula la discretización espacial de la pared basada en la velocidad de la luz y el tamaño del bin.

    # Se agrupan varios parámetros importantes en un diccionario para facilitar su uso posterior.
    params = {
        'cam_pixel_dim': cam_pixel_dim,  # Dimensión del arreglo de píxeles de la cámara.
        'camera_FOV': camera_FOV,  # Campo de visión total de la cámara.
        'camera_FOV_center': camera_FOV_center,  # Centro del campo de visión.
        'FOV_radius': FOV_radius,  # Tamaño de cada píxel en el FOV.
        'laser_intensity': laser_intensity,  # Intensidad del láser.
        'bin_size': bin_size,  # Tamaño de cada intervalo de tiempo (bin).
        'c': c,  # Velocidad de la luz.
        'laser_pos': laser_pos,  # Posición del láser.
        'laser_normal': laser_normal,  # Dirección del láser.
        'wall_discr': wall_discr,  # Discretización de la pared.
    }

    # ----- Cálculo del número de bins de tiempo para la simulación -----
    furthest_scene_point = np.array([xmax, ymax, zmax])  # Define el punto más lejano de la escena.
    furthest_spad_point = np.array([-params['camera_FOV'] / 2, -params['camera_FOV'], 0])  # Define el punto más lejano relativo al SPAD.
    d1 = np.linalg.norm(furthest_scene_point - laser_pos)  # Calcula la distancia desde el láser hasta el punto más lejano de la escena.
    d2 = np.linalg.norm(furthest_spad_point - furthest_scene_point)  # Calcula la distancia desde el punto más lejano de la escena hasta el SPAD.
    max_dist_travel = d1 + d2  # Suma ambas distancias para obtener la distancia máxima que recorre la señal.
    # Calcula el número de bins de tiempo necesarios, añadiendo un 20% extra como margen.
    num_time_bins = int(np.ceil((max_dist_travel / c / bin_size + 0.2 * max_dist_travel / c / bin_size)))
    params['num_time_bins'] = num_time_bins  # Almacena el número de bins en el diccionario de parámetros.

    # ----- Creación de paredes y techo (si no se solicitan ocultar) -----
    if not hide_walls:
        # Crea la pared de fondo usando el origen, el vector de ancho y el vector de altura.
        back_wall = create_wall_mesh(np.array([xmin, ymax, 0]), np.array([xmax - xmin, 0, 0]), np.array([0, 0, zmax]))
        # Crea la pared derecha.
        right_wall = create_wall_mesh(np.array([xmax, ymax, 0]), np.array([0, -ymax, 0]), np.array([0, 0, zmax]))
        # Crea la pared izquierda.
        left_wall = create_wall_mesh(np.array([xmin, ymax, 0]), np.array([0, -ymax, 0]), np.array([0, 0, zmax]))
        # Crea el techo.
        ceiling = create_wall_mesh(np.array([xmin, ymax, zmax]), np.array([xmax - xmin, 0, 0]), np.array([0, -ymax, 0]))
        back_wall = densify_mesh_if_needed(back_wall, min_triangles=MESH_MIN_TRIANGLES)
        right_wall = densify_mesh_if_needed(right_wall, min_triangles=MESH_MIN_TRIANGLES)
        left_wall = densify_mesh_if_needed(left_wall, min_triangles=MESH_MIN_TRIANGLES)
        ceiling = densify_mesh_if_needed(ceiling, min_triangles=MESH_MIN_TRIANGLES)
        objects.extend([back_wall, right_wall, left_wall, ceiling])  # Agrega las paredes y el techo a la lista de objetos.

    # ----- Configuración de la pared frontal dispersa -----
    x_start = xmin  # Define el inicio en X para la pared frontal.
    x_end = 0  # Define el fin en X para la pared frontal.
    z_start = zmin  # Define el inicio en Z para la pared frontal.
    z_end = zmax  # Define el fin en Z para la pared frontal.
    sphere_radius = 0.015  # Radio de cada esfera que compone la pared dispersa.
    spacing = 0.3  # Espaciado entre esferas.
    front_wall_color = [200, 200, 200, 255]  # Color (RGBA) de la pared frontal.
    front_wall_origin = np.array([xmin, ymin, zmin])  # Origen de la pared frontal.
    front_wall_width = np.array([x_end - x_start, 0, 0])  # Dimensiones en el ancho de la pared frontal.
    front_wall_height = np.array([0, 0, z_end - z_start])  # Dimensiones en la altura de la pared frontal.
    # Crea la pared frontal dispersa utilizando las esferas.
    front_wall_spheres = create_sparse_wall(origin=front_wall_origin, width_vec=front_wall_width, height_vec=front_wall_height, color=front_wall_color, sphere_radius=sphere_radius, spacing=spacing)

    # ----- Carga y posicionamiento de los objetos principales -----
    for obj_data in object_positions:
        obj_file = obj_data['obj_file']  # Nombre del archivo del objeto.
        xcoord = obj_data['xcoord']  # Coordenada X del objeto.
        ycoord = obj_data['ycoord']  # Coordenada Y del objeto.
        zcoord = obj_data['zcoord']  # Coordenada Z del objeto.
        w = obj_data['w']  # Escala o ancho deseado del objeto.
        yaw = obj_data['yaw']  # Rotación en torno al eje Z.
        pitch = obj_data['pitch']  # Rotación en torno al eje X.
        roll = obj_data['roll']  # Rotación en torno al eje Y.
        u = u = np.array([1, 0, 0]) # Vector de dirección para la rotación (eje X).
        v1 = np.array([xcoord, ycoord, zcoord])  # Vector de traslación para posicionar el objeto.
        theta = -np.clip(np.dot(u, v1) / (np.linalg.norm(u) * np.linalg.norm(v1)), -1, 1) # Calcula el coseno del ángulo entre el vector de dirección y el vector de traslación. (Foreshortening)

        # Verifica si el objeto proviene de archivos subidos (uploaded) o si ya existe en la carpeta 'objects'.
        if uploaded_objs and obj_file.startswith("uploaded_"):
            uploaded_file = uploaded_objs[obj_file]  # Obtiene el archivo subido.
            uploaded_file.seek(0)  # Reinicia la posición del archivo.
            obj = trimesh.load(uploaded_file, file_type='obj', force='mesh')  # Carga el objeto usando trimesh.
            if isinstance(obj, trimesh.Scene):  # Si el objeto es una escena, lo convierte a un mesh concatenado.
                obj = obj.dump(concatenate=True)
        else:
            obj_path = os.path.join(object_folder, obj_file)  # Construye la ruta completa al archivo del objeto.
            obj = trimesh.load(obj_path, force='mesh')  # Carga el objeto desde la ruta especificada.
        obj = densify_mesh_if_needed(obj, min_triangles=MESH_MIN_TRIANGLES)
        # Ajusta la escala del objeto para que tenga el tamaño deseado.
        obj_extents = obj.extents  # Obtiene las dimensiones actuales del objeto.
        scale_factors = [w / obj_extents[0], 1.1 / obj_extents[2]]  # Calcula factores de escala en X y Z.
        scale_factor = min(scale_factors)  # Selecciona el menor de los factores para mantener la proporción.
        obj.apply_scale(scale_factor)  # Aplica la escala al objeto.

        # Aplica las rotaciones (pitch, roll y yaw) al objeto.
        rotation = rotation_matrix(pitch, [1, 0, 0])  # Calcula la matriz de rotación para pitch (eje X).
        obj.apply_transform(rotation)  # Aplica la transformación de pitch.
        rotation_roll = rotation_matrix(roll, [0, 1, 0])  # Calcula la matriz de rotación para roll (eje Y).
        obj.apply_transform(rotation_roll)  # Aplica la transformación de roll.
        rotation_z = rotation_matrix(theta, [0, 0, 1])  # Calcula la matriz de rotación para yaw (eje Z).
        obj.apply_transform(rotation_z)  # Aplica la transformación de yaw.

        # Ajusta la posición vertical del objeto para que su base esté en z=0.
        z_min = obj.vertices[:, 2].min()  # Encuentra la coordenada mínima en Z del objeto.
        obj.apply_translation([0, 0, -z_min])  # Traslada el objeto para nivelar la base a z=0.
        obj.apply_translation(v1)  # Traslada el objeto a la posición deseada en (x, y, z).

        scene_objects.append(obj)  # Agrega el objeto transformado a la lista de objetos principales.

    if object_meshes:
        for mesh in object_meshes:
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            scene_objects.append(mesh.copy())

    objects.extend(scene_objects)  # Combina los objetos principales con los demás objetos de la escena.

    # ----- Creación de la escena 3D con trimesh -----
    scene = trimesh.Scene()  # Crea una nueva escena vacía.
    scene.add_geometry(objects)  # Agrega todos los objetos (paredes, techo, objetos) a la escena.
    front_rotation = rotation_matrix(np.radians(90), [1, 0, 0])  # Calcula una rotación de 90° alrededor del eje X.
    second_rotation = rotation_matrix(np.radians(180), [0, 0, 1])  # Calcula una rotación de 180° alrededor del eje Z.
    scene.apply_transform(front_rotation)  # Aplica la primera rotación a la escena.
    scene.apply_transform(second_rotation)  # Aplica la segunda rotación a la escena.

    # Agrega una esfera para representar la posición del láser en la escena.
    laser_sphere = trimesh.creation.icosphere(radius=0.04)  # Crea una esfera de radio 0.04.
    laser_sphere.apply_translation(laser_pos)  # Traslada la esfera a la posición del láser.
    scene.add_geometry(laser_sphere)  # Agrega la esfera a la escena.

    # ----- Configuración de los píxeles de la cámara -----
    cam_pixel_dim = params['cam_pixel_dim']  # Recupera la dimensión de la cámara (cantidad de píxeles por lado).
    # Genera las coordenadas en X para cada píxel dentro del FOV.
    pixel_x = np.linspace(
        params['camera_FOV_center'][0] - params['camera_FOV'] / 2 + params['camera_FOV'] / (2 * cam_pixel_dim),
        params['camera_FOV_center'][0] + params['camera_FOV'] / 2 - params['camera_FOV'] / (2 * cam_pixel_dim),
        cam_pixel_dim
    )
    # Genera las coordenadas en Y para cada píxel dentro del FOV.
    pixel_y = np.linspace(
        params['camera_FOV_center'][1] - params['camera_FOV'] / 2 + params['camera_FOV'] / (2 * cam_pixel_dim),
        params['camera_FOV_center'][1] + params['camera_FOV'] / 2 - params['camera_FOV'] / (2 * cam_pixel_dim),
        cam_pixel_dim
    )
    # Genera los tiempos correspondientes a cada bin de tiempo.
    pixel_t = np.linspace(0, (params['num_time_bins'] - 1) * params['bin_size'], params['num_time_bins'])

    # Crea una malla 2D con las coordenadas X e Y de los píxeles.
    X, Y = np.meshgrid(pixel_x, pixel_y)  # Crea matrices de coordenadas X e Y.
    cam_pos = np.vstack([X.ravel(), Y.ravel(), np.zeros(cam_pixel_dim**2)]).T  # Forma un arreglo con las posiciones 3D de cada píxel (con Z=0).

    # Crea una malla de índices para los píxeles (útil para mapear la posición en la matriz de la cámara).
    X_ind, Y_ind = np.meshgrid(range(params['cam_pixel_dim']), range(params['cam_pixel_dim']), indexing='xy')
    cam_pos_ind = np.vstack([X_ind.ravel(), Y_ind.ravel()]).T  # Vectoriza los índices de los píxeles.

    # Crea una malla 3D que incluye las coordenadas de los píxeles y el tiempo, simulando la variación temporal.
    X, Y, T = np.meshgrid(pixel_x, pixel_y, pixel_t)
    cam_pixel = np.vstack([X.ravel(), Y.ravel(), T.ravel()]).T  # Combina las coordenadas espaciales y temporales en un solo arreglo.

    # ----- Cálculo de la intensidad medida en cada píxel (simulación de rayos) -----
    combined_mesh = trimesh.util.concatenate(objects)  # Combina todos los objetos en un único mesh para facilitar la intersección.
    triangles = combined_mesh.triangles  # Extrae los triángulos del mesh combinado.
    triangle_normals = combined_mesh.face_normals  # Extrae las normales de cada triángulo.
    num_bins = params['num_time_bins']  # Número total de bins de tiempo.
    y_meas_vec = np.zeros(cam_pixel.shape[0])  # Inicializa un vector para almacenar la intensidad medida en cada píxel en cada bin.

    fourpi = 4 * np.pi * np.pi  # Constante usada para normalizar la intensidad.
    floor_normal = np.array([0, 0, 1])  # Vector normal para el piso, usado en el cálculo de la atenuación.

    # Se itera sobre cada triángulo del mesh para calcular su contribución a la medición.
    for idx in range(len(triangles)):
        triangle = triangles[idx]  # Obtiene el triángulo actual.
        normal = triangle_normals[idx]  # Obtiene la normal del triángulo actual.
        area = 0.5 * np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))  # Calcula el área del triángulo.
        scene_center = triangle.mean(axis=0)  # Calcula el centroide del triángulo.

        # Calcula la pendiente y la intersección para determinar la relación entre el centro del triángulo y los píxeles.
        m = (scene_center[1] - cam_pos[:, 1]) / (scene_center[0] - cam_pos[:, 0])  # Calcula la pendiente.
        b = scene_center[1] - np.dot(m, scene_center[0])  # Calcula la ordenada al origen.
        xint = -b / m  # Determina el punto de intersección en el eje X.

        noc = xint > 0  # Selecciona los píxeles que cumplen la condición (por ejemplo, que se encuentren frente al triángulo).
        if np.sum(noc) > 0:
            lps = params['laser_pos'] - scene_center  # Vector desde el láser hasta el centro del triángulo.
            fovsp = cam_pos[noc, :] - scene_center  # Vectores desde el centro del triángulo hasta los píxeles seleccionados.
            d1s = np.sum(lps**2)  # Calcula el cuadrado de la distancia del láser al centro del triángulo.
            d2s = np.sum(fovsp**2, axis=1)  # Calcula el cuadrado de la distancia del centro del triángulo a cada píxel.
            d1 = np.sqrt(d1s)  # Calcula la distancia real (d1).
            d2 = np.sqrt(d2s)  # Calcula la distancia real (d2).
            distance = d1 + d2  # Suma ambas distancias para obtener la distancia total recorrida.
            tbin = distance / (c * params['bin_size'])  # Determina el bin de tiempo basado en la distancia total.
            arrival_bin = np.ceil(tbin).astype(int)  # Redondea hacia arriba para obtener el bin de tiempo correspondiente.

            # Calcula factores de atenuación basados en la orientación entre la luz, la normal y las superficies.
            dot1 = np.maximum(0, np.sum(np.dot(normal, lps / d1)))  # Atenuación entre la normal del triángulo y el vector desde el láser.
            dot2 = np.maximum(0, np.sum(normal * fovsp / d2[:, np.newaxis], axis=1))  # Atenuación entre la normal y el vector hacia el píxel.
            dot3 = np.maximum(0, np.sum(np.dot(params['laser_normal'], -lps / d1)))  # Atenuación basada en la dirección del láser.
            dot4 = np.maximum(0, np.sum(floor_normal * -fovsp / d2[:, np.newaxis], axis=1))  # Atenuación basada en la normal del piso.

            # Calcula la intensidad que llega al píxel considerando la intensidad del láser, el área del triángulo y los factores de atenuación.
            intensity = params['laser_intensity'] * area * (dot1 * dot2 * dot3 * dot4) / (fourpi * d1s * d2s)

            # Determina la posición en el vector de mediciones donde se acumulará la intensidad calculada.
            #
            # IMPORTANT FIX:
            # cam_pos_ind starts at 0, so we must NOT subtract 1 from the pixel index.
            # Also, repeated indices can occur, so np.add.at is safer than y_meas_vec[coord] += intensity.
            arrival_bin_idx = arrival_bin - 1
            valid = (arrival_bin_idx >= 0) & (arrival_bin_idx < num_bins)

            if np.any(valid):
                pix_x = cam_pos_ind[noc, 0][valid]
                pix_y = cam_pos_ind[noc, 1][valid]
                t_idx = arrival_bin_idx[valid]
                intensity_valid = intensity[valid]

                # y_meas_vec is later reshaped as (y, x, t) with order='F':
                # flat_index = y + x*N + t*N*N
                coord = (
                    pix_y
                    + pix_x * params['cam_pixel_dim']
                    + t_idx * params['cam_pixel_dim']**2
                )

                np.add.at(y_meas_vec, coord, intensity_valid)  # Accumulate safely.

    # Reestructura el vector de mediciones a un arreglo 3D (dimensiones: píxeles x píxeles x bins de tiempo).
    y_meas_vec = y_meas_vec.reshape((params['cam_pixel_dim'], params['cam_pixel_dim'], num_bins), order='F') # Imagen transitoria 🐰

    # ----- Aplicación de ruido a la señal (si se activa) -----
    if add_noise:
        y_with_background = add_background_noise(y_meas_vec, sbr=SBR)  # Añade ruido ambiental (background).
        y_with_shot_noise = add_poisson_noise(y_with_background, scale_factor=poisson_scale_factor)  # Añade ruido de disparo (shot noise) usando una distribución Poisson.
        y_with_sensor_noise = add_sensor_noise(y_with_shot_noise, SNR_dB)  # Añade ruido propio del sensor basado en SNR.
        y_meas_vec_noisy = y_with_sensor_noise  # La medición final incluye todos los tipos de ruido.
    else:
        y_meas_vec_noisy = y_meas_vec  # Si no se activa el ruido, se utiliza la medición original sin modificar.
    y_meas_vec_noisy = np.roll(y_meas_vec_noisy, shift=-1, axis=-1)
    # ----- Configuración de la visualización 3D con Plotly (solamente para visualizar la escena modelada)-----
    fig3d = go.Figure()  # Crea una figura vacía en Plotly.

    # Agrega cada objeto (paredes, techo, etc.) a la figura 3D.
    for mesh in objects:
        vertices = mesh.vertices  # Obtiene los vértices del objeto.
        faces = mesh.faces  # Obtiene las caras (triángulos) del objeto.
        x, y, z = vertices.T  # Separa las coordenadas X, Y y Z de los vértices.
        i, j, k = faces.T  # Separa los índices de los triángulos.
        fig3d.add_trace(go.Mesh3d(
            x=x, y=y, z=z,  # Coordenadas de los vértices.
            i=i, j=j, k=k,  # Índices de las caras.
            color='blue',  # Color para representar paredes y techo.
            opacity=1.0,  # Opacidad completa.
            name='Walls & Ceiling'  # Nombre en la leyenda.
        ))

    # Agrega los objetos principales (modelos 3D) a la figura en un color diferente.
    for idx, obj in enumerate(scene_objects):
        fig3d.add_trace(go.Mesh3d(
            x=obj.vertices[:, 0],  # Coordenadas X de los vértices del objeto.
            y=obj.vertices[:, 1],  # Coordenadas Y.
            z=obj.vertices[:, 2],  # Coordenadas Z.
            i=obj.faces[:, 0],  # Índices de las caras.
            j=obj.faces[:, 1],
            k=obj.faces[:, 2],
            color='orange',  # Color distintivo para los objetos principales.
            opacity=1.0,  # Opacidad completa.
            name=f'Object {idx + 1}'  # Nombre en la leyenda.
        ))

    # Configura el layout de la figura 3D, definiendo títulos de ejes, posición de la cámara y dimensiones.
    fig3d.update_layout(
        scene=dict(
            xaxis_title='X',  # Título del eje X.
            yaxis_title='Y',  # Título del eje Y.
            zaxis_title='Z',  # Título del eje Z.
            aspectmode='data',  # Mantiene la proporción de los datos.
            camera=dict(
                eye=dict(x=-1.5, y=-1.5, z=1),  # Posición de la cámara.
                center=dict(x=0, y=0, z=0),  # Centro de la vista.
                up=dict(x=0, y=0, z=1)  # Dirección "arriba" para la cámara.
            )
        ),
        title="3D Scene Visualization",  # Título de la visualización.
        width=800,  # Ancho de la figura en píxeles.
        height=800  # Altura de la figura en píxeles.
    )

    # Se podría guardar la simulación en un archivo, pero estas líneas se encuentran comentadas.
    # filename = f"Simulacion_Con_Malla_{int(params['bin_size'] * 1e12)}ps.mat"
    # savemat(filename, {'params': params, 'y_meas_vec': y_meas_vec_noisy, 'objects': objects})

    # Devuelve la figura 3D, el vector de mediciones (con ruido si se añadió) y el diccionario de parámetros.
    return fig3d, y_meas_vec_noisy, params




# -------------------------------------------------------------------------
# REPLACEMENT BLOCK: put this after your simulation(...) function and
# remove/comment the old interactive visualization + old CRB block.
# -------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from trimesh.transformations import rotation_matrix

# -----------------------------
# Batch/output configuration
# -----------------------------
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OBJECT_DIR = ROOT / "objects"
FACET_PATH = OBJECT_DIR / "facet.obj"

OUTPUT_ROOT = ROOT / "crb_outputs"
OUTPUT_TRANSIENT_DIR = OUTPUT_ROOT / "time_compressed_png"
OUTPUT_SCENE_DIR = OUTPUT_ROOT / "facet_obj"
OUTPUT_MAP_DIR = OUTPUT_ROOT / "uncertainty_maps"

# To look like Fig. 3(b) in the paper, use "max".
# Use "sum" if you specifically want total integrated energy over time.
TIME_COMPRESSION = "max"   # options: "max" or "sum"

# You said diffuse albedo is always one and height is not estimated.
# Therefore the unknown parameter vector is psi = [rho, phi].
FACET_WIDTH = 0.50
FACET_BASE_Z = 0.0

# If your facet.obj is already vertical, change this to 0.0.
# In your original object_positions block you used pitch=1.57, so the
# default here keeps that convention.
FACET_BASE_PITCH = np.pi / 2
FACET_BASE_ROLL = 0.0

# After applying FACET_BASE_PITCH and FACET_BASE_ROLL, this is the normal
# direction of the facet face in its local pose. The paper uses
# n_s = [-cos(phi), -sin(phi), 0], i.e. the facet faces the occluding edge.
FACET_LOCAL_NORMAL_AFTER_BASE_ROT = np.array([-1.0, 0.0, 0.0])

SIMULATION_KWARGS = dict(
    xmin=-1.5,
    xmax=1.5,
    ymax=3.0,
    zmax=3.0,
    camera_FOV=0.5,
    cam_pixel_dim=64,
    bin_size=3.9e-10,
    laser_intensity=1000,
    hide_walls=True,
    SNR_dB=30,
    SBR=5,
    poisson_scale_factor=1,
    add_noise=False,
)

# CRB background rate. This is not "adding a noisy transient"; it is the
# expected background/dark-count rate in the Poisson model g = y + b.
# If you truly want no background, set this to 0.0, but keep eps in Fisher.
B_DARK_LEVEL = 0.01


def ensure_output_dirs():
    for folder in (OUTPUT_ROOT, OUTPUT_TRANSIENT_DIR, OUTPUT_SCENE_DIR, OUTPUT_MAP_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def safe_name(rho, phi_deg):
    return f"rho_{rho:.2f}_phi_{phi_deg:06.2f}".replace(".", "p")


def make_facet_mesh(rho, phi, width=FACET_WIDTH, facet_path=FACET_PATH):
    """Create one positioned facet mesh for a given polar location.

    The target location is:
        x = rho*cos(phi), y = rho*sin(phi), z = FACET_BASE_Z

    The target normal is:
        n_s = [-cos(phi), -sin(phi), 0]

    That is the convention used in the Active Corner Camera CRB paper.
    """
    mesh = trimesh.load(str(facet_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh = mesh.copy()

    if mesh.extents[0] <= 0:
        raise ValueError("facet.obj has zero extent in x; cannot scale by width.")
    mesh.apply_scale(width / mesh.extents[0])

    # Keep the same base orientation convention that you used in object_positions.
    if FACET_BASE_PITCH != 0.0:
        mesh.apply_transform(rotation_matrix(FACET_BASE_PITCH, [1, 0, 0]))
    if FACET_BASE_ROLL != 0.0:
        mesh.apply_transform(rotation_matrix(FACET_BASE_ROLL, [0, 1, 0]))

    # Rotate in azimuth so that [-1,0,0] becomes [-cos(phi), -sin(phi), 0].
    # Rz(phi) @ [-1,0,0] = [-cos(phi), -sin(phi), 0].
    mesh.apply_transform(rotation_matrix(phi, [0, 0, 1]))

    # Put base on z=0 after rotations.
    z_min = mesh.vertices[:, 2].min()
    mesh.apply_translation([0.0, 0.0, FACET_BASE_Z - z_min])

    # Move to polar location.
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    mesh.apply_translation([x, y, 0.0])

    return mesh


def export_facet_scene_obj(facet_mesh, output_path):
    """Export the positioned facet as an OBJ.

    I include a small laser sphere at the origin only for visual reference.
    It is not used in the CRB computation.
    """
    laser = trimesh.creation.icosphere(radius=0.04)
    laser.apply_translation([0.0, 0.0, 0.0])

    scene_mesh = trimesh.util.concatenate([facet_mesh.copy(), laser])
    scene_mesh.export(str(output_path))


def simulate_facet_signal_expected(rho, phi, width=FACET_WIDTH):
    """Expected noiseless transient y(rho, phi) for the positioned facet."""
    facet_mesh = make_facet_mesh(rho, phi, width=width)

    _, y_signal, _ = simulation(
        **SIMULATION_KWARGS,
        object_positions=[],
        object_meshes=[facet_mesh],
    )
    return y_signal


def forward_g_polar(rho, phi, width=FACET_WIDTH):
    """Poisson rate g = y + b.

    y is the expected transient from the facet.
    b is a constant background/dark rate.
    """
    y_signal = simulate_facet_signal_expected(rho, phi, width=width)
    return y_signal + B_DARK_LEVEL


def forward_g_from_psi(psi, width=FACET_WIDTH):
    rho, phi = np.asarray(psi, dtype=float)
    return forward_g_polar(rho, phi, width=width).ravel()


def finite_difference_jacobian_polar(psi0, steps, width=FACET_WIDTH):
    psi0 = np.asarray(psi0, dtype=float)
    steps = np.asarray(steps, dtype=float)

    if psi0.shape != (2,):
        raise ValueError("This version estimates only psi = [rho, phi].")
    if steps.shape != psi0.shape:
        raise ValueError(f"psi0 shape {psi0.shape} and steps shape {steps.shape} differ.")
    if np.any(steps <= 0):
        raise ValueError(f"All finite-difference steps must be positive: {steps}")

    g0_arr = forward_g_polar(psi0[0], psi0[1], width=width)
    g0 = g0_arr.ravel()
    J = np.zeros((g0.size, psi0.size), dtype=float)

    for m in range(psi0.size):
        dpsi = np.zeros_like(psi0)
        dpsi[m] = steps[m]

        psi_plus = psi0 + dpsi
        psi_minus = psi0 - dpsi

        # rho must remain positive.
        if psi_minus[0] <= 0:
            raise ValueError(
                f"Finite-difference step for rho is too large: "
                f"rho0={psi0[0]}, step={steps[0]}, rho_minus={psi_minus[0]}"
            )

        g_plus = forward_g_from_psi(psi_plus, width=width)
        g_minus = forward_g_from_psi(psi_minus, width=width)

        J[:, m] = (g_plus - g_minus) / (2.0 * steps[m])

    return g0_arr, J


def fisher_poisson(g, J, eps=1e-12):
    """Fisher matrix for independent Poisson measurements.

    I[m,k] = sum_q (1/g_q) * (dg_q/dpsi_m) * (dg_q/dpsi_k)
    """
    g = np.asarray(g, dtype=float).ravel()
    J = np.asarray(J, dtype=float)
    g_safe = np.maximum(g, eps)
    return J.T @ ((1.0 / g_safe)[:, None] * J)


def crb_region_physical_polar(rho0, phi0, CRB, k=3.0, n_points=300):
    """Convert the [rho, phi] covariance to an ellipse in physical x-y space,
    then convert the ellipse back to polar coordinates for plotting.
    """
    Sigma_rhophi = CRB[np.ix_([0, 1], [0, 1])]

    A = np.array(
        [
            [np.cos(phi0), -rho0 * np.sin(phi0)],
            [np.sin(phi0),  rho0 * np.cos(phi0)],
        ]
    )
    Sigma_xy = A @ Sigma_rhophi @ A.T
    mu_xy = np.array([rho0 * np.cos(phi0), rho0 * np.sin(phi0)])

    eigenvalues, eigenvectors = np.linalg.eigh(Sigma_xy)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    t = np.linspace(0.0, 2.0 * np.pi, n_points)
    unit_circle = np.vstack([np.cos(t), np.sin(t)])
    axes_lengths = k * np.sqrt(eigenvalues)

    xy_curve = mu_xy[:, None] + eigenvectors @ np.diag(axes_lengths) @ unit_circle

    x_curve = xy_curve[0, :]
    y_curve = xy_curve[1, :]
    rho_curve = np.sqrt(x_curve**2 + y_curve**2)
    phi_curve = np.arctan2(y_curve, x_curve)

    return rho_curve, phi_curve


def compute_crb_and_region(rho0, phi0, width=FACET_WIDTH, steps=None, k=3.0):
    if steps is None:
        steps = np.array([0.01, np.deg2rad(0.1)], dtype=float)

    psi0 = np.array([rho0, phi0], dtype=float)
    g0_arr, J = finite_difference_jacobian_polar(psi0, steps=steps, width=width)

    I = fisher_poisson(g0_arr, J)
    CRB = np.linalg.pinv(I)

    sigma_rho = np.sqrt(max(CRB[0, 0], 0.0))
    sigma_phi = np.sqrt(max(CRB[1, 1], 0.0))

    rho_curve, phi_curve = crb_region_physical_polar(
        rho0=rho0,
        phi0=phi0,
        CRB=CRB,
        k=k,
    )

    return {
        "g0_arr": g0_arr,
        "J": J,
        "I": I,
        "CRB": CRB,
        "sigma_rho": sigma_rho,
        "sigma_phi": sigma_phi,
        "rho_curve": rho_curve,
        "phi_curve": phi_curve,
    }


def time_compress_transient(y_signal, mode=TIME_COMPRESSION):
    """Convert a 3D transient [y, x, t] into one 2D image."""
    if mode == "max":
        image = np.max(y_signal, axis=2)
    elif mode == "sum":
        image = np.sum(y_signal, axis=2)
    else:
        raise ValueError("TIME_COMPRESSION must be 'max' or 'sum'.")

    # Keep your previous visual shift if you want the same orientation.
    image = np.roll(image, shift=1, axis=1)
    return image


def save_time_compressed_png(y_signal, output_path, title, mode=TIME_COMPRESSION):
    image = time_compress_transient(y_signal, mode=mode)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(image, origin="lower", cmap="hot", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("SPAD pixel x")
    ax.set_ylabel("SPAD pixel y")
    fig.colorbar(im, ax=ax, label=f"{mode} over time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def print_crb_report(rho_i, phi_deg, result):
    print(
        f"rho={rho_i:.3f} m, phi={phi_deg:.1f} deg | "
        f"sigma_rho={result['sigma_rho']:.6g} m | "
        f"sigma_phi={result['sigma_phi']:.6g} rad "
        f"({np.rad2deg(result['sigma_phi']):.6g} deg)"
    )


def main():
    ensure_output_dirs()

    crb_steps = np.array([0.01, np.deg2rad(0.1)], dtype=float)

    ranges = np.array([0.5, 1.0, 1.5])
    angles_deg = np.array([30, 60, 90, 120, 150])
    angles = np.deg2rad(angles_deg)

    fig, ax = plt.subplots(figsize=(7, 5), subplot_kw={"projection": "polar"})

    print("\n=== Running CRB sweep and exporting artifacts ===")
    for rho_i in ranges:
        for phi_i, phi_deg in zip(angles, angles_deg):
            name = safe_name(rho_i, phi_deg)

            # 1) Compute CRB using the full transient, not the compressed PNG.
            result = compute_crb_and_region(
                rho0=rho_i,
                phi0=phi_i,
                width=FACET_WIDTH,
                steps=crb_steps,
                k=3.0,
            )
            print_crb_report(rho_i, phi_deg, result)

            # 2) Save time-compressed nominal measurement PNG.
            # Remove the dark level so the PNG represents the facet signal y, not y+b.
            y_nominal = np.maximum(result["g0_arr"] - B_DARK_LEVEL, 0.0)
            png_path = OUTPUT_TRANSIENT_DIR / f"{name}_{TIME_COMPRESSION}.png"
            save_time_compressed_png(
                y_nominal,
                png_path,
                title=fr"$\rho={rho_i:.2f}$ m, $\phi={phi_deg:.1f}^\circ$",
                mode=TIME_COMPRESSION,
            )

            # 3) Save positioned facet scene as OBJ.
            facet_mesh = make_facet_mesh(rho_i, phi_i, width=FACET_WIDTH)
            obj_path = OUTPUT_SCENE_DIR / f"{name}_facet_scene.obj"
            export_facet_scene_obj(facet_mesh, obj_path)

            # 4) Add uncertainty bubble to final polar map.
            ax.plot(result["phi_curve"], result["rho_curve"], linewidth=1.5)
            ax.scatter([phi_i], [rho_i], s=12)

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(0, 2.0)
    ax.set_title(r"Conditional CRB regions for $\psi=[\rho,\phi]$, $k=3$")

    map_path = OUTPUT_MAP_DIR / "crb_standard_regions.png"
    fig.tight_layout()
    fig.savefig(map_path, dpi=200)
    plt.close(fig)

    print("\nDone.")
    print(f"Time-compressed PNGs: {OUTPUT_TRANSIENT_DIR}")
    print(f"Facet OBJ scenes:      {OUTPUT_SCENE_DIR}")
    print(f"CRB map:               {map_path}")


if __name__ == "__main__":
    main()