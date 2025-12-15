from fastapi import FastAPI
from pydantic import BaseModel
from keras.layers import TFSMLayer
import tensorflow as tf
class ReviewRequest(BaseModel):
    review: str

app = FastAPI()

model = TFSMLayer(
    "model_attention_full", 
    call_endpoint="serving_default"
)

@app.post("/predict")
def predict_sentiment(req: ReviewRequest):
    inp = tf.constant([[req.review]])
    out = model(inp, training=False)

    # out — це dict {'output_0': tensor(...)} або сам тензор
    if isinstance(out, dict):
        out = out["output_0"]

    score = float(out.numpy()[0][0])
    sentiment = "positive" if score >= 0.5 else "negative"

    return {
        "score": score,
        "sentiment": sentiment
    }


# @app.post("/predict")
# def predict_sentiment(req: ReviewRequest):
#     input_batch = [[req.review]]
#     output = model(input_batch, training=False).numpy()
#     score = float(output[0][0])
#     sentiment = "positive" if score >= 0.5 else "negative"
#     return {"score": score, "sentiment": sentiment}


# @app.post("/predict")
# def predict(data: ReviewRequest):
#     print(">>> GOT REVIEW:", data.review)

#     out = model({"input_1": tf.constant([data.review])})
#     print(">>> MODEL OUTPUT:", out)

#     return {"pred": str(out)}
