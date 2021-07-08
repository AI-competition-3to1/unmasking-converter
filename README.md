# unmasking-converter
Unmasking converter web site project 


# Init Directory

`app/` Web service에 사용되는 model 관련 패키지
    - `public/` public source 관련
    - `server/` server 관련  
    - `source/` source 관련 

`models/` 학습에 사용되는 model 관련 패키지
    - `data/` 학습에 사용되는 dataset 디렉토리 
        - `face_verification_accessories/` 
    - `masking/`  원본 이미지에 마스크 착용 이미지 학습
    - `unmasking/` 마스크 착용 이미지에서 미착용 원본 추출
