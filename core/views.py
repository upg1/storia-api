from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.http import JsonResponse, StreamingHttpResponse
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate

from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
import gensim
from gensim.models import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS as gensim_stopwords
nltk.download('stopwords')
import json
import os
import openai
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from apify_client import ApifyClient
from dotenv import load_dotenv

import uuid 


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"
PINECONE_CREDENTIALS = os.getenv('PINECONE_API_KEY')
HUGGING_FACE = os.getenv("HUGGING_FACE_KEY")
pc = Pinecone(api_key=PINECONE_CREDENTIALS)
index = pc.Index("storia")

# Langchain pinecone vector store:

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)



vectorstore = PineconeVectorStore(index_name="storia", embedding=embeddings)




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
        # An id tying the tweet to the request that created it. 
        metadata['request_id'] = uuid.uuid4()


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

@api_view(['POST'])

all_stopwords = set(nltk.corpus.stopwords.words('english')).union(gensim_stopwords).union({'medtwitter', 'https', 'www', 'medx'})


def lemmatize(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in all_stopwords and len(token) > 3:
            result.append(lemmatize(token))
    return result

def postTopics(posts):
    processed_posts = [preprocess(post) for post in posts]
    dictionary = Dictionary(processed_posts)
    corpus = [dictionary.doc2bow(text) for text in processed_posts]
    lda_model = LdaMulticore(corpus, num_topics=10, id2word=dictionary, passes=10, workers=2)
    topics_for_posts = []
    for text in corpus:
        post_topics = sorted(lda_model.get_document_topics(text), key=lambda x: x[1], reverse=True)
        top_3_topics = post_topics[:3]
        actual_topics = [' '.join(word for word, _ in lda_model.show_topic(topic[0], topn=3)) for topic in top_3_topics]
        topics_for_posts.append(actual_topics)
    return topics_for_posts


def query_pinecone(query_embedding, str_handles):

    if (len(str_handles) == 0):
        results = index.query(
                    vector=query_embedding, 
                    top_k=10, 
                    include_metadata=True
        )  
    else:
        handles = [str_handles]
        print("handles", handles)
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

    try:
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
        created_at = metadata["created_at"]
        #  = metadata["transcript_chunk"]
     
        scores.append(score)
        chosen_tweets.append(["tweet: " + tweet, "handle: " + handle, "created_at: " + handle])

        max_similarity = max(scores)
        print("MAX SIMILIARITY:", max_similarity)

        return chosen_tweets
    except:
        print("No relevant tweets")
        return "failure"

@api_view(['POST'])
def answer_query(request):
    user_query = json.loads(request.body)
    print("User Query: ", user_query) #EX: {'user_query': 'tell me about tumors', 'handles_mentioned': ['sama', elonmusk]}
    chosen_tweets = retrieve_tweet(user_query)
    print(chosen_tweets)

    if (chosen_tweets == "failure"):
        return JsonResponse("No relevant tweets found. Please try another query!", safe=False)

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

def fetchPostsByHandles(handle_list):
	apify_client = ApifyClient(APIFY_APP_TOKEN)
	run_input = {
	    "handles": handle_list,
	    "tweetsDesired": 100,
	    "addUserInfo": True,
	    "startUrls": [],
	    "proxyConfig": { "useApifyProxy": True },
	}


	# Start an actor and wait for it to finish
	actor_call = apify_client.actor('quacker/twitter-scraper').call(run_input=run_input)
	items = []
	for item in apify_client.dataset(run["defaultDatasetId"]).iterate_items():
		metadata = {}
	    metadata['screen_name'] = item['user']['screen_name']
	    metadata['full_text'] = item['full_text']
	    metadata['url'] = item['url']
	    metadata['created_at'] = item['created_at']
	    # An id tying the tweet to the request that created it. 
	    metadata['request_id'] = uuid.uuid4()
		items.append(Document(page_content=item.full_text, metadata=metadata))

	vectorstore.from_documents(items, embedding=embeddings)
	return items




def huggingface_chat_with_system_prompt(system_prompt_str, human_prompt_str):

	default_system_prompt_str = """You are a journalist and must use the following tweets 
	to answer the question at the end by constructing a narrative about the timeline of the tweets. 
	use tweet metadata like created_at to establish the timeline, and in each step of the timeline, 
	specify two or three sentences explaining that trending tweet. 

	{context}

	Question: {question}

	Helpful Answer:"""
	custom_rag_prompt = PromptTemplate.from_template(default_system_prompt_str)

	retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
	rag_chain = (
	    {"context": retriever | format_docs, "question": RunnablePassthrough()}
	    | custom_rag_prompt
	    | llm
	    | StrOutputParser()
	)

	res = rag_chain.invoke(human_prompt_str)
	print(res.content)
	return res.content

# @api_retrieve(['GET'])
def get_tweets_by_handles(user_query):
    handle_list = json.loads(request.body)

    hundredPosts = fetchPostsByHandles(handle_list)

    return JsonResponse(hundredPosts, safe=False)

@api_view(['POST'])
def system_prompt_test(request):
    user_query = json.loads(request.body)
    print(chosen_tweets)
    # ADD LLM call and pass in chosen Tweets.
    # Example structure of chosen_tweets: [["tweet1", "@handle1"], ["tweet2", "@handle2"]]
    chosen_tweets_str = "\n".join([f'{tweet}: {handle}' for tweet, handle in chosen_tweets])
    prompt = "Please use the following tweets to give the user a summary about their query. Please cite the twitter handles and only use the ones that are presented as handles. " + chosen_tweets_str



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

    for message in chat_completion:
        print(message.choices[0].delta.content, end="")

    return JsonResponse("completed request", safe=False)


