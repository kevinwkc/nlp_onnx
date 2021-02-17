import mlflow.pyfunc
import pandas as pd

data = pd.DataFrame({'txt':["I love this bakery so much and even though it's a little far from my house I will still drive to come here! ", "How About a rating of ZERO! This outfit is a total scam. I ordered a Valentines bouquet for my wife on Feb. 9 to be delivered Feb. 13. It Never showed up."]})

model_name = "mybert"
model_version = 24

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

print(model.predict(data))