import transformers


"""모델 새로 만들때나 쓰는 api"""
config = transformers.PretrainedConfig(
    model_type="bert",
    # 모델 어휘 사전 크기, 모델이 인식할 수 있는 고유 토큰의 수
    vocab_size= 30522,

    # 멀티헤드 어텐션에서 사용되는 어텐션 헤드 수
    num_attention_heads=12,
    # 모든 어텐션값 출력 여부
    output_attentions=True,

    # 모델 은닉계층 노드수, 클수록 표현력이 좋아짐
    hidden_size=768,
    # 은닉 계층수, 클수록 표현력이 좋아짐
    num_hidden_layers=12,
    # 은닉계층 상태 출력 여부
    output_hidden_states=True,
    # 은닉계층 활성화 함수
    hidden_act="gelu",
    # 드롭아웃확률
    hidden_dropout_prob=0.1,

    # 내부 차원 크기
    intermediate_size=3072,
    # 가중치 초기범위 설정
    initializer_range=0.02,
)


