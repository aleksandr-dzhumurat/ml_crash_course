CURRENT_DIR = $(shell pwd)
USER_NAME = $(shell whoami)
RANDOM_PORT = $(shell expr 1000 + $$RANDOM % 9999)

build-network:
	docker network create ds_network -d bridge || true

build: build-network
	docker build \
        -f ${CURRENT_DIR}/Dockerfile \
        -t ds-container:dev ${CURRENT_DIR}

train:
	docker run -it --rm --network=ds_network \
	    -p 8892:8888 \
		-e CONFIG_PATH="/srv/src/config.yml" \
	    -v "${CURRENT_DIR}/src:/srv/src" \
	    -v "${CURRENT_DIR}/data:/srv/data" \
	    --name ${USER_NAME}_ml_docker \
	    ds-container:dev train

serve:
	docker run -it --rm --network=ds_network \
	    -p 5000:5000 \
		-e CONFIG_PATH="/srv/src/config.yml" \
	    -v "${CURRENT_DIR}/src:/srv/src" \
	    -v "${CURRENT_DIR}/data:/srv/data" \
	    --name ${USER_NAME}_ml_docker \
	    ds-container:dev serve

labelstudio:
	docker run -it -d --rm --network=ds_network \
	    -p 8080:8080 \
	    -v "${CURRENT_DIR}/src:/srv/src" \
	    -v "${CURRENT_DIR}/data:/srv/data" \
	    --name labelstudio_ml_docker \
	    ds-container:dev labelstudio

notebook:
	docker run -it -d --rm --network=ds_network \
	    -p 8888:8888 \
	    -v "${CURRENT_DIR}/src:/srv/src" \
	    -v "${CURRENT_DIR}/data:/srv/data" \
	    --name jupyter_ml \
	    ds-container:dev notebook

load:
	sudo docker run -it --rm --network=ds_network \
	    -p 8892:8888 \
	    -v "${CURRENT_DIR}/src:/srv/src" \
	    -v "${CURRENT_DIR}/data:/srv/data" \
	    --name ${USER_NAME}_ml_docker \
	    ds-container:dev load

stop:
	sudo docker rm -f ${USER_NAME}_notebook