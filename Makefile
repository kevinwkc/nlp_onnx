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
	export MLFLOW_TRACKING_URI=http://localhost:5000

	export MLFLOW_S3_ENDPOINT_URL=https://s3.tor01.cloud-object-storage.appdomain.cloud
	export AWS_PROFILE=ibm

	export MYSQL_DATABASE="mydb" MYSQL_USER="flow" MYSQL_PASSWORD="flow" MYSQL_ROOT_PASSWORD="password"

	mkdir {artifact-root,data}
#https://mlflow-single-core-cluster-community.innosre-managed-586fba9d8cb47b239a7531fe80d39153-0000.us-south.containers.appdomain.cloud/

run:
	#-rm -rf roberta-saved
	mlflow run . --experiment-name RoBERTa | tee run.log

predict:
	mlflow run . -e predict

predict_local: export
	mlflow models predict -m runs:/8073aee631f94fae9e587a556ca1b798/model --input-path input.csv --content-type csv

	#mlflow models predict -m models:/mybert/5 --input-path input.csv --content-type csv

serve: export
	mlflow models serve -m runs:/8073aee631f94fae9e587a556ca1b798/model -p 1234

	#mlflow models serve -m models:/mybert/5 -p 1234


server:
	#https://medium.com/@moyukh_51433/mlflow-storing-artifacts-in-hdfs-and-in-an-sqlite-db-7be26971b6ab
	#docker pull larribas/mlflow
	docker run -d --rm -p 5000:5000 -v /d/temp/nlp_onnx/artifact-root:/artifact-root -v /d/temp/nlp_onnx/db:/db --name flow-server larribas/mlflow --backend-store-uri sqlite:////db/mlflow.db --default-artifact-root file:////tmp/artifact-root --host 0.0.0.0 -p 5000

ui:
	mlflow ui --backend-store-uri sqlite:///db/mlflowl.db --default-artifact-root /d/temp/nlp_onnx/artifact-root -h 0.0.0.0 -p 5000
#--backend-store-uri sqlite:///d/temp/nlp_onnx/db/mlflowl.db --default-artifact-root file:///d/temp/nlp_onnx/artifact-roo
sdebug:
	docker exec -it flow-server /bin/bash

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