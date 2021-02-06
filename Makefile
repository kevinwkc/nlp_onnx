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