from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import json
import os

import openai
from pinecone import Pinecone, ServerlessSpec

EMBEDDING_MODEL = "text-embedding-ada-002"
PINECONE_CREDENTIALS = os.getenv('PINECONE_CREDENTIALS')
pc = Pinecone(api_key=PINECONE_CREDENTIALS)

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

@api_upsert(['POST'])
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
        

        # Assuming you need to do something with metadata and tweet here,
        # for demonstration, we're just appending it to a response list.
        response_data.append({'metadata': metadata, 'tweet': tweet})
    
    # Returning the processed data as JSON response
    return JsonResponse(response_data, safe=False)

 def postTopics(posts):
 	pass

@api_upsert(['GET'])
def get_top_topics(request):

	





