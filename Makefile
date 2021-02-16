img:
	docker build -t senti_v docker

debug:
	docker run -it --rm --entrypoint /bin/bash -v D:\temp\models\text\machine_comprehension\roberta\production\app_vol:/data  senti_v

play:
	docker exec -it my-senti_v /bin/bash

api:
	docker run -d -v D:\temp\models\text\machine_comprehension\roberta\production\app_vol:/data -p 8080:8080 --name my-senti_v senti_v

onnx:
	docker run -p 8888:8888 --rm -v /${PWD}/:/scripts/data  --name my-onnx onnx/onnx-ecosystem

cuda:
	#docker pull nvidia/cuda:11.2.0-runtime
	docker run -it --rm --name my-cuda nvidia/cuda:11.2.0-runtime

export:
	export MLFLOW_S3_ENDPOINT_URL=https://s3.tor01.cloud-object-storage.appdomain.cloud
	export AWS_PROFILE=ibm
	export MLFLOW_TRACKING_URI=http://localhost:5000
	export MYSQL_DATABASE="mydb" MYSQL_USER="flow" MYSQL_PASSWORD="flow" MYSQL_ROOT_PASSWORD="password"
#https://mlflow-single-core-cluster-community.innosre-managed-586fba9d8cb47b239a7531fe80d39153-0000.us-south.containers.appdomain.cloud/

run:
	mlflow run . | tee run.log

server:
	#docker pull larribas/mlflow
	docker run -d --rm -p 5000:5000 --name flow-server larribas/mlflow --host 0.0.0.0

up:
	MYSQL_DATABASE="mydb" MYSQL_USER="flow" MYSQL_PASSWORD="flow" MYSQL_ROOT_PASSWORD="password" docker-compose up -d --build

dn:
	docker-compose down

mydb:
	#docker run -d -p 3306:3306 -e MYSQL_ROOT_PASSWORD=root mysql/mysql-server:5.7.28
	#docker run -it -p 3306:3306 --name mydb -e MYSQL_ROOT_PASSWORD=root  mysql/mysql-server:5.7.28
	docker run -d -p 3306:3306 --name mydb -v /tmp/mysql:/var/lib/mysql -e MYSQL_ROOT_PASSWORD=root  mysql/mysql-server:5.7.28
#	docker run -d -p 3306:3306 --name mydb -v "D:\temp\nlp_onnx\db":/var/lib/mysql -e MYSQL_ROOT_PASSWORD=root  mysql/mysql-server:5.7.28
	docker logs -f mydb

clean:
	-docker kill mydb
	-docker rm mydb
	cd db && rm -rf ./*
	rm -rf /tmp/mysql/*