from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_http_methods
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric, remove_stopwords, stem_text, STOPWORDS
import re
import json
import os
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv("secrets.env")

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
PINECONE_CREDENTIALS = os.getenv('PINECONE_API_KEY')
HUGGING_FACE = os.getenv("HUGGING_FACE_KEY")
pc = Pinecone(api_key=PINECONE_CREDENTIALS)
index = pc.Index("storia")

from openai import OpenAI

client = OpenAI(
	base_url="https://hnkyim7dr0wn0en3.us-east-1.aws.endpoints.huggingface.cloud/v1/", 
	api_key= HUGGING_FACE 
)


"""
Embedding / Pinecone related
"""
def get_embedding(text, model=EMBEDDING_MODEL):
    result = openai.embeddings.create(
      model=model,
      input=text
    )
    print("gone through")
    # return result["data"][0]["embedding"]
    return result.data[0].embedding

@api_view(['POST'])
def upsert_tweets(request):
    # Assuming the body of the request is the raw JSON data you need to process
    data_json = json.loads(request.body)
    response_data = []

    data_json = data_json['data']
    
    for tweet_data in data_json:
        metadata = {}
        tweet = tweet_data.get('full_text', '')

        # Extracting the required data
        user_data = tweet_data.get('user', {})
        metadata['user'] = user_data.get('name', '')
        metadata['screen_name'] = user_data.get('screen_name', '')
        metadata['full_text'] = tweet
        metadata['url'] = f"https://twitter.com/{user_data.get('screen_name', '')}/status/{tweet_data.get('id_str', '')}"
        metadata['created_at'] = tweet_data.get('created_at', '')

        tweet_embedding = get_embedding(tweet)
        tweet_id = tweet_data.get('id_str', '')

        tweet_data = [{
            "id": f"{tweet_id}",
            "values": tweet_embedding,
            "metadata": {
                "screen_name": metadata['screen_name'],
                "tweet_text": metadata['full_text'],
                "created_at": metadata['created_at'],
                "url": metadata['url'],
                "user": metadata['user']
            }
        }]
        print("Upserting title embedding data")
        index.upsert(tweet_data)

        # # Assuming you need to do something with metadata and tweet here,
        # # for demonstration, we're just appending it to a response list.
        response_data.append({'metadata': metadata, 'tweet': tweet})
    
    # Returning the processed data as JSON response
    return JsonResponse(response_data, safe=False)

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)

def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)

additional_stopwords = {"medtwitt", "s", "medx", "type", "common","medtwitter","⁉","🩸","heart🥰","isn","t"}
all_stopwords = STOPWORDS.union(additional_stopwords)

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in all_stopwords])

def preprocess(post):
    post = remove_urls(post)
    post = remove_emojis(post)
    post = ' '.join(preprocess_string(post, [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_numeric]))
    post = remove_stopwords(post)
    return post.split()

def postTopics(posts):
    processed_posts = [preprocess(post) for post in posts]
    dictionary = Dictionary(processed_posts)
    corpus = [dictionary.doc2bow(text) for text in processed_posts]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)

    topics_for_posts = []
    for text in corpus:
        post_topics = lda_model.get_document_topics(text)
        actual_topics = [(lda_model.show_topic(topic[0], topn=5), topic[1]) for topic in post_topics]
        topics_for_posts.append(actual_topics)

    return topics_for_posts


def query_pinecone(query_embedding, str_handles):
    handles = [str_handles]
    print("handles", handles)

    if (len(handles) == 0):
        results = index.query(
                    vector=query_embedding, 
                    top_k=10, 
                    include_metadata=True
        )  
    else:
        results = index.query(
                vector=query_embedding, 
                top_k=10, 
                filter = {
                        "screen_name": {"$in": handles}
                },
                include_metadata=True
            )  
    return [(match["id"], match["score"], match["metadata"]) for match in results["matches"]]

# @api_retrieve(['GET'])
def retrieve_tweet(user_query):
    query = user_query["user_query"]
    query_embedding = get_embedding(query)

    get_tweets = query_pinecone(query_embedding, user_query["handles_mentioned"])

    print("\n\n")
    print("______________________________________________________\n") 
    print(f"Successfully retreived: {len(get_tweets)} sources from search")

    scores = []
    chosen_tweets = []

    for tweet_item in get_tweets:

        chunk_identifier, score, metadata = tweet_item[0], tweet_item[1], tweet_item[2]
        print(f"Chunk Identifier: {chunk_identifier}")
        print(f"Score: {score}")
        print(f"Metadata: {metadata}")

        tweet = metadata["tweet_text"]
        handle = metadata["screen_name"]
        #  = metadata["transcript_chunk"]
     
        scores.append(score)
        chosen_tweets.append(["tweet: " + tweet, "twitter handle: " + handle])

    max_similarity = max(scores)
    print("MAX SIMILIARITY:", max_similarity)

    return chosen_tweets

@api_view(['POST'])
def answer_query(request):
    user_query = json.loads(request.body)
    print("User Query: ", user_query) #EX: {'user_query': 'tell me about tumors', 'handles_mentioned': ['sama', elonmusk]}
    chosen_tweets = retrieve_tweet(user_query)
    print(chosen_tweets)

    # Example structure of chosen_tweets: [["tweet1", "@handle1"], ["tweet2", "@handle2"]]
    chosen_tweets_str = "\n".join([f'{tweet}: {handle}' for tweet, handle in chosen_tweets])
    prompt = "Please use the following tweets to give the user a concise summary about their query. Please answer in one short form summary, no bullet points with the twitter handle given printed at the end. " + chosen_tweets_str

    # LLM call (Mistral 7B model)
    chat_completion = client.chat.completions.create(
        model="tgi",
        messages=[
        {
            "role": "user",
            "content": user_query["user_query"] 
        },
        {
            "role": "assistant",
            "content": prompt
        },
    ],
        stream=True,
        max_tokens=500
    )

    # for message in chat_completion:
    #     print(message.choices[0].delta.content, end="")

    final_response_content = ""
    for message in chat_completion:
        final_response_content += message.choices[0].delta.content
        print(message.choices[0].delta.content, end="")

    return JsonResponse(final_response_content, safe=False)


