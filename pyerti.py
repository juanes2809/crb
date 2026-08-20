import os
import sys
from pathlib import Path
from utils.check_overlaps import check_overlaps
from utils.create_wall import create_wall_mesh
from utils.get_files import get_obj_files
from utils.sparse_wall import create_sparse_wall
from utils.densify_mesh import densify_mesh_if_needed


ROOT = Path(__file__).resolve().parent
SIM_ROOT = ROOT / "NLOS-Simulator"
UTILS_DIR = ROOT / "utils"
object_folder = SIM_ROOT / "objects"

np = None
trimesh = None
rotation_matrix = None
create_wall_mesh = None
add_background_noise = None
add_poisson_noise = None
add_sensor_noise = None
create_sparse_wall = None


def _load_simulation_dependencies():
    global np
    global trimesh
    global rotation_matrix
    global create_wall_mesh
    global add_background_noise
    global add_poisson_noise
    global add_sensor_noise
    global create_sparse_wall

    if str(UTILS_DIR) not in sys.path:
        sys.path.append(str(UTILS_DIR))

    try:
        import numpy as numpy_module
        import trimesh as trimesh_module
        from trimesh.transformations import rotation_matrix as rotation_matrix_function
        from create_wall import create_wall_mesh as create_wall_mesh_function
        from noise import add_background_noise as add_background_noise_function
        from noise import add_poisson_noise as add_poisson_noise_function
        from noise import add_sensor_noise as add_sensor_noise_function
        from sparse_wall import create_sparse_wall as create_sparse_wall_function
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Instala las dependencias de simulacion: pip install numpy trimesh plotly"
        ) from exc

    np = numpy_module
    trimesh = trimesh_module
    rotation_matrix = rotation_matrix_function
    create_wall_mesh = create_wall_mesh_function
    add_background_noise = add_background_noise_function
    add_poisson_noise = add_poisson_noise_function
    add_sensor_noise = add_sensor_noise_function
    create_sparse_wall = create_sparse_wall_function

    return go


c = 299792458
ymin = 0
zmin = 0

xmin = -1.5
xmax = 1.5
ymax = 3
zmax = 3

camera_FOV = 0.25
cam_pixel_dim = 64
bin_size = 3.9e-10
laser_intensity = 1000
hide_walls = False
SNR_dB = 30
SBR = 5
poisson_scale_factor = 1000
add_noise = True
MESH_MIN_TRIANGLES = 5000

# Posiciones de objetos (debes tener los .obj en la carpeta 'objects/')
object_positions = [
    {
        'obj_file': 'bunny.obj',
        'xcoord': 0.6,
        'ycoord': 1.25,
        'zcoord': 0,
        'w': 0.5,
        'yaw': 0,
        'pitch': 1.57,
        'roll': 0,
    }
]

def simulation(xmin, xmax, ymax, zmax, camera_FOV, cam_pixel_dim, bin_size, laser_intensity, object_positions, hide_walls, SNR_dB, SBR, poisson_scale_factor, add_noise, uploaded_objs=None):
    go = _load_simulation_dependencies()

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
    objects.extend(front_wall_spheres)

    # ----- Carga y posicionamiento de los objetos principales -----
    for obj_data in object_positions:
        if "mesh" in obj_data:
            obj = obj_data["mesh"].copy()
            obj = densify_mesh_if_needed(obj, min_triangles=MESH_MIN_TRIANGLES)
            scene_objects.append(obj)
            continue
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
            coord = np.dot(arrival_bin - 1, params['cam_pixel_dim']**2) + np.dot((cam_pos_ind[noc, 0] - 1), params['cam_pixel_dim']) + cam_pos_ind[noc, 1]
            y_meas_vec[coord] += intensity  # Acumula la intensidad en el vector de mediciones.

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

    # Corrige el orden espacial de columnas: la ultima columna del frame colapsado
    # debe aparecer como primera columna. En el cubo transitorio esto corresponde
    # al segundo eje espacial, no al eje temporal.
    y_meas_vec_noisy = np.roll(y_meas_vec_noisy, shift=1, axis=1)
    params['columns_rolled'] = True

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
