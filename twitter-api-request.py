import requests, os, pandas
from requests_oauthlib import OAuth1
from urllib.parse import parse_qs

REQUEST_TOKEN_URL = "https://api.twitter.com/oauth/request_token"
AUTHORIZE_URL = "https://api.twitter.com/oauth/authorize?oauth_token="
ACCESS_TOKEN_URL = "https://api.twitter.com/oauth/access_token"

CONSUMER_KEY = ""
CONSUMER_SECRET = ""

OAUTH_TOKEN = ""
OAUTH_TOKEN_SECRET = ""

REQUIRED_FIELDS = ['id','id_str','screen_name','location','description','url',
                   'followers_count','friends_count', 'listed_count', 'created_at',
                   'favourites_count', 'verified','statuses_count','lang','status',
                   'default_profile','default_profile_image','has_extended_profile',
                   'name','bot']

def setup_oauth():
    # Request token
    oauth = OAuth1(CONSUMER_KEY, client_secret=CONSUMER_SECRET)
    r = requests.post(url=REQUEST_TOKEN_URL, auth=oauth)
    credentials = parse_qs(r.content)

    resource_owner_key = credentials.get('oauth_token')[0]
    resource_owner_secret = credentials.get('oauth_token_secret')[0]

    # Authorize
    authorize_url = AUTHORIZE_URL + resource_owner_key
    print ('Please go here and authorize: ' + authorize_url)

    verifier = input('Please input the verifier: ')
    oauth = OAuth1(CONSUMER_KEY,
                   client_secret=CONSUMER_SECRET,
                   resource_owner_key=resource_owner_key,
                   resource_owner_secret=resource_owner_secret,
                   verifier=verifier)

    # Finally, Obtain the Access Token
    r = requests.post(url=ACCESS_TOKEN_URL, auth=oauth)
    credentials = parse_qs(r.content)
    token = credentials.get('oauth_token')[0]
    secret = credentials.get('oauth_token_secret')[0]
    return token, secret


def get_oauth():
    oauth = OAuth1(CONSUMER_KEY,
                client_secret=CONSUMER_SECRET,
                resource_owner_key=OAUTH_TOKEN,
                resource_owner_secret=OAUTH_TOKEN_SECRET)
    return oauth

def write_to_csv(responses):
    user_l = []
    for response in responses:
        final_fields = {}

        try:
            response_dict = response.json()[0]
        except AttributeError:
            response_dict = response

        for key, value in response_dict.items():
            if key in REQUIRED_FIELDS:
                final_fields[key] = value
        user_l.append(final_fields)

        user_df = pandas.DataFrame(user_l)
    print(user_df)

    twitter_user_data = open(os.getcwd() + '/twitterUsers_final.csv', 'w')
    user_df.to_csv(twitter_user_data,index=False)
    twitter_user_data.close()

def fix_requests(r):
    import json
    return json.loads(r.content.decode('utf-8'))

if __name__ == "__main__":
    if not OAUTH_TOKEN:
        token, secret = setup_oauth()
        print ("OAUTH_TOKEN: " + token)
        print ("OAUTH_TOKEN_SECRET: " + secret)
    else:
        oauth = get_oauth()

       # for searching using screen name
        screen_name = ['justinbieber','potus']
        responses = []
        for name in screen_name:
            response = requests.get(url="https://api.twitter.com/1.1/users/lookup.json?screen_name="+str(name), auth=oauth)
            if response.status_code == 200:
                responses.append(response)
        write_to_csv(responses)

'''
        # for a general search
        query = 'bot'
        for page in range(1,10):
            responses = requests.get(url="https://api.twitter.com/1.1/users/search.json?q="+str(query)+"&page="+str(page), auth=oauth)
            if responses.status_code == 200:
                fixed_r = fix_requests(responses)
                write_to_csv(fixed_r)
'''
