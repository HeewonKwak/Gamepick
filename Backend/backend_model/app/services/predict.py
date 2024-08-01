# model
# import openai
import google.generativeai as genai
from core.config import API_KEY
# data
import numpy as np
from services.filters import filter
from services.preprocess import * 


class HybridModel():
    def __init__(self, user_data):
        self.user_games_id = user_data.games
        self.age = int(user_data.age)
        self.platform = user_data.platform
        self.players = int(user_data.players)
        self.major_genre = user_data.major_genre
        self.tag = tag_preprocessing(user_data.tag)

        self.initialize_data()
        self.preprocess()
    
    def initialize_data(self):
        self.game_table = load_data_from_redis('game')
        self.model_table = load_data_from_redis('ease')
        self.user_table = load_data_from_redis('cf_model')
    
    def preprocess(self):
        # model_table preprocess
        # user_idx로 묶어서 id를 배열로 합치기
        self.user_table = self.user_table.groupby('user_idx')['id'].apply(list).reset_index()
        self.user_table["id"] = self.user_table["id"].apply(lambda x: np.array(x, dtype=int))
        
    def predict(self):
        if not self.user_games_id:
            return []
        
        self.user_table['similarity'] = self.user_table['id'].apply(lambda x: game_similarity(x, self.user_games_id))

        similarity_df = self.user_table[self.user_table['similarity'] == max(self.user_table['similarity'])]
        similarity_df = self.model_table[self.model_table['user'].isin(select_similar_user_idx(similarity_df))]

        df_extracted = self.game_table[self.game_table['id'].isin(list(similarity_df['item']))]
        df_extracted = df_extracted.set_index('id')
        df_extracted = df_extracted.loc[list(similarity_df['item'])]
        df_extracted = df_extracted.reset_index()

        final_id = filter(df_extracted, self.age, self.platform, self.players, self.major_genre, 'cf')
        return list(final_id)[:10]
    
class HybridModel_Modify():
    def __init__(self):
        self.initialize_data()
        self.preprocess()
    
    def initialize_data(self):
        self.game_table = load_data_from_redis('game')
        self.model_table = load_data_from_redis('ease')
        self.user_table = load_data_from_redis('cf_model')
    
    def preprocess(self):
        # model_table preprocess
        # user_idx로 묶어서 id를 배열로 합치기
        self.user_table = self.user_table.groupby('user_idx')['id'].apply(list).reset_index()
        self.user_table["id"] = self.user_table["id"].apply(lambda x: np.array(x, dtype=int))
        
    def predict(self, user_data):
        self.user_games_id = user_data.games
        self.age = int(user_data.age)
        self.platform = user_data.platform
        self.players = int(user_data.players)
        self.major_genre = user_data.major_genre
        self.tag = tag_preprocessing(user_data.tag)

        if not self.user_games_id:
            return []
        
        self.user_table['similarity'] = self.user_table['id'].apply(lambda x: game_similarity(x, self.user_games_id))

        similarity_df = self.user_table[self.user_table['similarity'] == max(self.user_table['similarity'])]
        similarity_df = self.model_table[self.model_table['user'].isin(select_similar_user_idx(similarity_df))]

        df_extracted = self.game_table[self.game_table['id'].isin(list(similarity_df['item']))]
        df_extracted = df_extracted.set_index('id')
        df_extracted = df_extracted.loc[list(similarity_df['item'])]
        df_extracted = df_extracted.reset_index()

        final_id = filter(df_extracted, self.age, self.platform, self.players, self.major_genre, 'cf')
        return list(final_id)[:10]

class Most_popular_filter():
    def __init__(self, user_data):
        self.age = int(user_data.age)
        self.platform = user_data.platform
        self.players = int(user_data.players)
        self.major_genre = user_data.major_genre

        self.initialize_data()
        self.preprocess_input()
        
    def initialize_data(self):
        self.game_table = load_data_from_redis('game')
        self.details_table = load_data_from_redis('details')

    def preprocess_input(self):
        # 필터링
        self.idx = filter(self.game_table, self.age, self.platform, self.players, self.major_genre, 'cb')

    def predict(self):
        self.details_table = self.details_table[self.details_table['id'].isin(self.idx)]
        self.details_table = self.details_table.sort_values(by="critic_score", ascending=False)
        return list(self.details_table.head(10)['id'])

import random

def get_random_adjective():
    """
    추천 게임에 대한 다양한 형용사를 무작위로 반환하는 함수
    """
    adjectives = ["exciting", "challenging", "fun", "unique", "immersive", "new"]
    return random.choice(adjectives)

def gemini_ai(user_data):
    """
    Gemini API를 이용하여 사용자에게 맞춤형 게임을 추천하는 함수

    Args:
        user_data: 사용자 정보를 담은 객체

    Returns:
        추천된 게임 목록 (문자열)
    """
    # set api key
    genai.configure(api_key=API_KEY)
    
    # 프롬프트 생성 (다양한 경우의 수 고려)
    prompt = f"I'm looking for {get_random_adjective()} games that is similar to the ones I've played. Considering my age ({user_data.age}), platform ({user_data.platform}), and preferred genre ({user_data.major_genre}), please suggest 10 games that I might enjoy."

    if user_data.tag:
        prompt += f" Additionally, I'm interested in games with {user_data.tag} elements."

    if user_data.players:
        prompt += f" I prefer games that can be played with {user_data.players} players."

    # 플레이한 게임 목록 추가
    prompt += f" I have already played: {user_data.games}. Please suggest new games that I haven't played yet."
    
    # 출력 예시
    prompt += f" Output format: 1. game1 2. game2 3. game3 4. game4 5. game5 6. game6 7. game7 8. game8 9. game9 10. game10"

    # Gemini 모델 설정
    generation_config = genai.GenerationConfig(
        temperature=0.7,  # 창의성 조절
        max_output_tokens=200,  # 출력 길이 조절
    )

    model = genai.GenerativeModel('gemini-pro', generation_config=generation_config)

    # API 호출 및 오류 처리
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error occurred: {e}")
        return "An error occurred while generating recommendations. Please try again later."