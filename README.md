# Proyecto 3 Hackaboss

### Instalación

Para poder trabajar con este repositorio, es necesario tener la versión 3.11.4 de Python y tener Git instalado en el sistema operativo. Se deben de seguir estos pasos para la correcta instalación:

- Primero se clona este repositorio de forma local en la ubicación deseada.
```sh
git clone https://github.com/cuicuidev/project_hackaboss

```
Esto va a generar una carpeta proyect_hackaboss que contiene todos los archivos y carpetas del proyecto.

- Después, hay que crear el entorno virtual con el siguiente comando:
```sh
python -m venv .env

```

- Deben además desactivar el entorno de Anaconda en caso de tenerlo instalado.
```sh
conda deactivate

```

- Acto seguido, activan el entorno recien instalado.
```sh
source ./.env/bin/activate

```

- Por último, instalan todas las dependencias del proyecto en el entorno virtual.
```sh
pip install -r requirements.txt

```

Ya están listos para trabajar!!!