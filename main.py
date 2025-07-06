# Hugging Face의 'pipeline' 기능을 가져옵니다.
# 이 기능은 복잡한 모델 로딩, 토크나이징, 예측 과정을 몇 줄의 코드로 간단하게 만들어줍니다.
from transformers import pipeline

# --- 1. 긍정/부정 감정 분석 모델 테스트 ---
# 모델 이름: distilbert-base-uncased-finetuned-sst-2-english
print("="*50)
print("1. 긍정/부정 분석 모델 (distilbert-base-uncased-finetuned-sst-2-english)")
print("="*50)

# 'text-classification' 파이프라인을 생성하고, 사용할 모델을 지정합니다.
# 모델이 로컬에 없으면 Hugging Face 허브에서 자동으로 다운로드됩니다. (처음 실행 시 시간이 걸릴 수 있습니다)
sentiment_classifier = pipeline(
    "text-classification", 
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# --- 2. 6가지 다중 감정 분석 모델 테스트 ---
# 모델 이름: j-hartmann/emotion-english-distilroberta-base
print("\n" + "="*50)
print("2. 6가지 감정 분석 모델 (j-hartmann/emotion-english-distilroberta-base)")
print("="*50)

# 같은 방식으로 6가지 감정 분류 모델을 위한 파이프라인을 생성합니다.
emotion_classifier = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True # 모든 감정 점수를 확인하려면 True로 설정
)

# --- 3. 보고서에 사용할 문장들 테스트 ---
# 보고서에서 탐구하고자 했던 문장들을 리스트로 만듭니다.
test_sentences = [
    "I'm fine.",
    "I'm fine...", # 참고: 대부분의 모델은 '...'을 일반 마침표로 처리할 수 있습니다.
    "I'm fine, but I feel empty.",
    "This is an amazing experience!",
    "I hate this, it's a disaster.",
    "Human-inspired AI can detect the consumers’ subconscious emotions.", # 보고서의 핵심 문장
    "I am not happy." # 부정문 테스트
]

print("\n--- 분석 결과 ---\n")

# 각 문장을 반복하며 두 모델의 분석 결과를 출력합니다.
for sentence in test_sentences:
    print(f"▶ 원문: \"{sentence}\"")

    # 1번 모델 (긍정/부정) 결과 분석
    sentiment_result = sentiment_classifier(sentence)
    # 결과 형식: [{'label': 'POSITIVE', 'score': 0.999...}]
    print(f"  - 긍정/부정 분석 결과: {sentiment_result[0]['label']} (점수: {sentiment_result[0]['score']:.4f})")
    
    # 2번 모델 (6가지 감정) 결과 분석
    emotion_results = emotion_classifier(sentence)
    # 결과 형식: [[{'label': 'sadness', 'score': 0.9...}, {'label': 'joy', 'score': 0.0...}, ...]]
    # 가장 높은 점수를 받은 감정을 찾습니다.
    best_emotion = max(emotion_results[0], key=lambda x: x['score'])
    print(f"  - 6가지 감정 분석 결과: {best_emotion['label']} (점수: {best_emotion['score']:.4f})")
    
    print("-" * 20)