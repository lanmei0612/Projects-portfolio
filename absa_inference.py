import pandas as pd
from setfit import AbsaModel

df = pd.read_csv("reviews.csv", sep='\t')


def absa(inputs):
    data = {
        "review_id": [],
        "dish": [],
        "sentiment": []
    }

    model = AbsaModel.from_pretrained(
        "models/setfit-absa-model-aspect",
        "models/setfit-absa-model-polarity",
        spacy_model="en_core_web_lg"
    )

    predictions = model.predict(inputs)

    food_terms = ['chicken', 'duck', 'fish', 'beef', 'pork', 'rice', 'noodle',
                  'soup', 'sauce', 'wings', 'neck', 'meat', 'roll', 'taco',
                  'dumpling', 'ramen', 'dim sum', 'bubble tea', 'curry', 'Ice Cream']

    for i, predict in enumerate(predictions):
        for sentiment in predict:
            span = sentiment['span'].lower()
            if any(term in span for term in food_terms):
                data['review_id'].append(i)
                data['dish'].append(sentiment['span'])
                data['sentiment'].append(sentiment['polarity'])

    return pd.DataFrame(data)


results = absa(df['Review'])
print(results)
