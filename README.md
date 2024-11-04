# Modelos1
Proyecto sustituto para la asignatura de Modelos 1

## Fase 1
Abrir el archivo de la carpeta *Fase 1* en Google Colab y ejecutar las lineas secuencialmente. Tener en cuenta que se requiere un *kaggle.json* correspondiente a la cuenta de Kaggle para poder descargar los archivos de la competencia a utilizar.

## Fase 2
Para hacer uso de esta sección debemos tener instalado Docker; inicialmente copiar el repositorio localmente, y abrir una consola de comandos al interior de la carpeta *Fase 2*.

A continuación ingresar la siguiente linea de comando para empezar la creación de la imagen a utilizar:
```
docker build -t name -f Dockerfile.txt .
```

Posteriormente, ingresar la siguiente linea de comando para crear el contenedor a partir de la imagen anterior (IMAGE_ID -> hash/id de la imagen):
```
docker run -t -d IMAGE_ID
```

Finalmente, ingresar al contenedor utilizando la siguiente linea de comando (CONTAINER_ID -> hash/id del contenedor):
```
docker exec -it CONTAINER_ID /bin/bash
```

Podemos observar los archivos y la carpeta 'scripts' al interior utilizando el comando 'ls', así podemos ejecutar el archivo 'use_scripts.py' para realizar la prueba de funcionamiento de los scripts utilizando la siguiente linea (NOTA: El proceso de entrenamiento y predicción puede demorar un tiempo mayor cuando no se utiliza GPU):
```
python3 use_scripts.py
```

## Fase 3
Para hacer uso de esta sección debemos tener instalado Docker; inicialmente copiar el repositorio localmente, y abrir una consola de comandos al interior de la carpeta *Fase 3*.

A continuación ingresar la siguiente linea de comando para empezar la creación de la imagen a utilizar (**ADVERTENCIA:** La creación de la imagen puede demorar ~20 minutos):
```
docker build -t name -f Dockerfile.txt .
```

Posteriormente, ingresar la siguiente linea de comando para crear el contenedor a partir de la imagen anterior (IMAGE_ID -> hash/id de la imagen):
```
docker run -t -d IMAGE_ID
```

Finalmente, ingresar al contenedor utilizando la siguiente linea de comando (CONTAINER_ID -> hash/id del contenedor):
```
docker exec -it CONTAINER_ID /bin/bash
```

Podemos observar los archivos y la carpeta 'scripts' al interior utilizando el comando 'ls', ahora podemos ejecutar el archivo 'rest_api.py' para iniciar el servidor local utilizando la siguiente linea (NOTA: El proceso de entrenamiento y predicción puede demorar un tiempo mayor cuando no se utiliza GPU):
```
python3 rest_api.py
```

Una vez inicializado el servidor, abrimos otra terminal y volvemos a ingresar al contenedor, así podemos simular el envio de peticiones al servidor mediante el archivo 'client.py' utilizando el siguiente comando:
```
python3 client.py
```

