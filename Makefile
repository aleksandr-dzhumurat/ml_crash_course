CURRENT_DIR = $(shell pwd)
USER_NAME = $(shell whoami)
RANDOM_PORT = $(shell expr 1000 + $$RANDOM % 9999)

build-network:
	docker network create ds_network -d bridge || true

build: build-network
	docker build \
        -f ${CURRENT_DIR}/Dockerfile \
        -t ds-container:dev ${CURRENT_DIR}

run:
	docker run -it --rm --network=ds_network \
	    -p 8892:8888 \
	    -v "${CURRENT_DIR}/src:/srv/src" \
	    -v "${CURRENT_DIR}/data:/srv/data" \
	    --name ${USER_NAME}_ml_docker \
	    ds-container:dev train

labelstudio:
	docker run -it --rm --network=ds_network \
	    -p 8080:8080 \
	    -v "${CURRENT_DIR}/src:/srv/src" \
	    -v "${CURRENT_DIR}/data:/srv/data" \
	    --name labelstudi_ml_docker \
	    ds-container:dev labelstudio

load:
	sudo docker run -it --rm --network=ds_network \
	    -p 8892:8888 \
	    -v "${CURRENT_DIR}/src:/srv/src" \
	    -v "${CURRENT_DIR}/data:/srv/data" \
	    --name ${USER_NAME}_ml_docker \
	    ds-container:dev load

stop:
	sudo docker rm -f ${USER_NAME}_notebook