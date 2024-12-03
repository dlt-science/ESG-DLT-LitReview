# Brat labelling tool
# ===================

The brat labelling tool is a web-based tool for text annotation; that is, for adding notes to existing text documents. The basic idea behind brat is that text annotation should be easy and generalized enough to support many different annotation types.

To execute the brat labelling tool, we use the docker container from [Digital Humanities Lab](https://hub.docker.com/r/dhlabbasel/brat). We use the following command to start the container and mount the data and configuration volumes. The username, password and email are set to the default values of brat. The container is started in detached mode and the port 80 is mapped to the host system:

```bash

docker run --name=brat -d -p 80:80 -v /your/path/brat-data:/bratdata -v /your/path/brat-cfg:/bratcfg -e BRAT_USERNAME=brat -e BRAT_PASSWORD=brat -e BRAT_EMAIL=brat@example.com dhlabbasel/brat

```

If you are using a Apple M1 chip, you can use the following command:

```bash
docker run --name=brat -d -p 80:80 -v brat-data:/bratdata -v brat-cfg:/bratcfg -e BRAT_USERNAME=brat -e BRAT_PASSWORD=brat -e BRAT_EMAIL=brat@example.com --platform linux/amd64 dhlabbasel/brat
```

Make sure to enable `Use Rosetta for x86/amd64 emulation on Apple Silicon` in the Docker Desktop settings: `Preferences > Features in development`.

The command above creates two volumes, one for the data and one for the configuration. The data volume is used to store the annotated documents and the configuration volume is used to store the configuration files for the annotation types. The volumes are created with the following commands:

```bash
docker volume create --name brat-data
docker volume create --name brat-cfg
``` 
