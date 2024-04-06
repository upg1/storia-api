from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
import os
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
PINECONE_CREDENTIALS = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_CREDENTIALS)
index = pc.Index("storia")

"""
Embedding / Pinecone related
"""
def get_embedding(text, model=EMBEDDING_MODEL):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    print("gone through")
    return result["data"][0]["embedding"]

@api_view(['POST'])
def upsert_tweets(request):
    # Assuming the body of the request is the raw JSON data you need to process
    data_json = json.loads(request.body)
    response_data = []
    
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

def postTopics(posts):
 	pass


def query_pinecone(query_embedding, twitter_handle):
    if (len(twitter_handle) == 0):
        results = index.query(
                vector=query_embedding, 
                top_k=10, 
                include_metadata=True
            )  
    else:
        results = index.query(
                # namespace=ns, 
                vector=query_embedding, 
                top_k=10, 
                filter = {
                        "screen_name": {"$eq": twitter_handle}
                },
                include_metadata=True
            )  
    return [(match["id"], match["score"], match["metadata"]) for match in results["matches"]]

# @api_retrieve(['GET'])
def retrieve_tweet(user_query):
    query_embedding = get_embedding(user_query)

    get_tweets = query_pinecone(query_embedding)

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

        tweet = metadata["full_text"]
        #  = metadata["transcript_chunk"]
     
        scores.append(score)
        chosen_tweets.append(tweet)

    max_similarity = max(scores)
    print("MAX SIMILIARITY:", max_similarity)

    return chosen_tweets

@api_view(['POST'])
def answer_query(request):
    user_query = request.user_query
    chosen_tweets = retrieve_tweet(user_query)

    # ADD LLM call and pass in chosen Tweets.





