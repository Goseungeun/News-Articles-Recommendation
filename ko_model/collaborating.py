from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

import pandas as pd
import warnings
import csv

from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

#크롤링 데이터 받아오기
crawling_data = pd.read_csv('./output/Article_economy_201701_201804.csv', header=None, names=['publish_year', 'catagory', 'publish', 'title', 'content', 'url'])

#사용자가 기사에 점수 매긴 데이터 만들기 (임의로 만든거 -> 좋아요, 스크랩 기준으로 데이터 다시 생성해야함)
# f = open('./output/rating.csv','w',encoding='utf-8',newline='')
# rt = csv.writer(f)
# rt.writerow([1,'중국 한국산 배터리 전기차에 보조금 중단',30])
# rt.writerow([1,'카카오톡 신년 인사 폭주에 40분간 ‘불통’',10])
# rt.writerow([2,'중국 한국산 배터리 전기차에 보조금 중단',55])
# rt.writerow([2,'이코노 브리핑 삼성 분리세탁가능 ‘플렉스워시’ 공개 외',15])
# rt.writerow([2,'네이버 새해 푸드윈도 판매자 4000명으로...온라인 신선식품 시장 다크호스',60])
# rt.writerow([3,'현대차 작년 유럽 판매 역대 최다',25])
# rt.writerow([3,'이코노 브리핑 삼성 분리세탁가능 ‘플렉스워시’ 공개 외',15])
# rt.writerow([3,'자라 창업주 가로수길 HM건물 샀다',30])
# rt.writerow([4,'이코노 브리핑 삼성 분리세탁가능 ‘플렉스워시’ 공개 외',35])
# rt.writerow([4,'카카오톡 신년 인사 폭주에 40분간 ‘불통’',40])

# f.close()

# f = open('./output/rating.csv','a',encoding='utf-8',newline='')
# wr = csv.writer(f)
# wr.writerow([5,'만약 크롤링 한 기사에 없는 제목이라면',5])

# f.close()

#사용자가 기사에 매긴 점수 데이터 불러오기
rating_data = pd.read_csv('./output/rating.csv',header=None,names=['user','title','rate'])
# 크롤링 데이터중 title column만 사용
crawling_data = crawling_data[['title']]
#크롤링 데이터와 rating 데이터 title 기준으로 합치기
user_article_rating = pd.merge(rating_data,crawling_data,on = 'title')
#pivot table 생성 data = 점수, index = title, columns = user
user_article_rating = user_article_rating.pivot_table('rate',index = 'title',columns='user')

#pivot table의 Nan 값 0으로 바꿔줌
user_article_rating.fillna(0,inplace=True)
#기사별로 코사인 유사도 계산
article_based_collabor = cosine_similarity(user_article_rating)
article_based_collabor = pd.DataFrame(data = article_based_collabor, index = user_article_rating.index, columns=user_article_rating.index)
#title 들어가면 비슷한 점수를 받은 기사를 출력
def get_article_based_collabor(title):
    if title in article_based_collabor:
        recommend = article_based_collabor[title].sort_values(ascending=False)[:10]
        # 유사도가 너무 낮으면 제외
        except_low_value = recommend.values > 0.3
        # 유사도가 1은 자기 자신이므로 제외
        except_input_title = recommend.values < 1
        recommend_news = recommend[except_input_title & except_low_value]
        
    else : 
        #pivot table에 입력한 title의 기사 없을 때 추천 뉴스 없음
        recommend_news = '추천 뉴스가 없습니다.'
        
    return recommend_news

result = get_article_based_collabor('“신사업 등에 3조원 투자” SK이노베이션 5년간 1200명 채용')
#결과 값 출력
print(result)