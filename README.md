# Bank of China News Sentiment Monitor

ISOM5240 group project: **AI-Powered News Sentiment Monitoring for Bank of China Using Deep Learning**.

**Live app:** https://finnewssentiment-dfdyvtwwuydzxiend7o2zc.streamlit.app/

**Fine-tuned model:** https://huggingface.co/glindahuangsg/boc-finsentiment-distilbert

## What the app does

The app reads English financial news about Bank of China, finds the negative items, gives each one a priority, and suggests which team should handle it.

```
News headline
  → Pipeline 1: sentiment (negative / neutral / positive)
  → Pipeline 2: topic (one of 20 financial topics)
  → Alert rule: negative with confidence ≥ 0.85 becomes an alert;
                high priority if the topic is Legal | Regulation, Earnings,
                Financials or Treasuries | Corporate Debt, or if the news is about Bank of China;
                the topic sets the suggested team
```

| Pipeline | Model | Notes |
|---|---|---|
| 1 · Sentiment | [`glindahuangsg/boc-finsentiment-distilbert`](https://huggingface.co/glindahuangsg/boc-finsentiment-distilbert) | DistilBERT fine-tuned by the group on 9,450 financial sentences and headlines |
| 2 · Topic | [`leonas5555/finnews-topic-single-classify`](https://huggingface.co/leonas5555/finnews-topic-single-classify) | Published BERT classifier with 20 topics, used as it is |

Both models are loaded by name from Hugging Face when the app starts, and run on CPU.

## Pages

| Page | What it shows |
|---|---|
| Dashboard | Key numbers, sentiment mix, sentiment over time, negative items by topic |
| News search | Search the scored news, open the original article, download as CSV |
| Alert centre | Alerts by priority and suggested team; mark them reviewed or dismissed; change the threshold and priority topics |
| Report | A short summary of the news and alerts, to download |
| Analyse a text | Paste one headline and see its sentiment, topic and suggested team |
| Model check | Upload a labelled CSV and see the accuracy of the sentiment model |

News sources (sidebar): live Google News headlines for "Bank of China", an uploaded CSV, or demo headlines. The **About** filter keeps only the news about Bank of China itself.


## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first start downloads both models from Hugging Face (a few hundred MB). No token is needed, because both models are public.

## Check the deployed model

1. Download `app_test_sample.csv` from this repository (open the file, then click **Download raw file**).
2. Open the app and go to **Model check**.
3. Upload `app_test_sample.csv`.
4. The page shows the accuracy (0.8967, 269 of 300), the same as notebook 03, and a table of where the model is right and wrong.

The model never saw these 300 rows during training: they come from the held-out test set.

## Notes

This is an academic prototype for ISOM5240 at HKUST. It is not affiliated with Bank of China. The output is a screening aid for human review, not investment or compliance advice.
