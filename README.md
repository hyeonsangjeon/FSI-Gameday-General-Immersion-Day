# AWS FSI General Immersion Day for the AIML Gameday
AWS FSI General Immersion Day for AIML 

## Workshop 기본 실습 자료  
https://catalog.us-east-1.prod.workshops.aws/workshops/f3a3e2bd-e1d5-49de-b8e6-dac361842e76/ko-KR/preparation-guide/20-event-engine



### 실습 계정정보  
#### - https://bit.ly/3Oaocx8
- 실습 시작 후 오픈합니다. 


#### PuTTY 다운로드
윈도우 PC에서 핸즈온 과정 중 EC2 SSH 연동을 위해 사용합니다.
- downloadlink
https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html
- bin 64-bit x86:: https://the.earth.li/~sgtatham/putty/latest/w64/putty.exe



## [Compute EC2 optional 과정] AIML General Immersion Day


## EC2에 Model API 서버 인스턴스 생성하기
### Model-API Serving
Amazon Linux 2 인스턴스를 시작하고, 분류 모델 API를 생성해봅니다.  
- 딥러닝 모델은 사전에 생성되어있습니다. 
- EC2에 기동한 Flask API를 이용해서 NLP 한글문장 감정 분석 API를 호출해봅니다.  
![screenshot1](https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day/blob/main/pic/simple_architecture.png?raw=true)

### 1.EC2생성 
- 모델은 DeepLearning Tenserflow Bert Model로 구현되어있습니다. API는 모델을 Load한 뒤 Client Request를 호출하면 Predict 함수를 call합니다. 
- 다소 무거운 모델 API입니다. t2.xlarge(4CPU / 16G) 또는 GPU EC2를 권장합니다.
- API는 리눅스 Nvidia GPU서버의 경우 자동 GPU 모드를 사용합니다.
![screenshot2](https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day/blob/main/pic/step_0.png?raw=true)
![screenshot3](https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day/blob/main/pic/step_1.png?raw=true)

- VPC는 Network 실습에서 생성한 VPC-Lab을 선택합니다. 
![screenshot4](https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day/blob/main/pic/step_2.png?raw=true)

- ML Model API는 컨테이너 이미지 용량이 약 7Gb입니다. EBS의 크기를 50GB로 변경합니다.  
![screenshot5](https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day/blob/main/pic/step_3.png?raw=true)
- EC2인스턴스를 찾기 쉽도록 Tag이름(EC2's alias)을 기입합니다. 
![screenshot6](https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day/blob/main/pic/step_4.png?raw=true)

- 보안그룹에서 test할 API 의 TCP port를 바인딩 합니다. 8080 포트를 사용하며, 소스는 '내 IP'를 선택하여 client PC에서만 접속 가능하도록 선택합니다. 
![screenshot7](https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day/blob/main/pic/step_5.png?raw=true)

- EC2 인스턴스에 접속하기 위한 RSA 키 페어는 GID 실습에서 사용한 기존 키페어를 선택합니다.   
![screenshot8](https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day/blob/main/pic/step_6.png?raw=true)

### 2. Model API 사전 작업
- Linux EC2인스턴스에 진입합니다.
![screenshot8](https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day/blob/main/pic/step_7.png?raw=true)
```bash
ssh -i "AWS-ImmersionDay.pem" ec2-user@{퍼블릭 IPv4 주소}
```

- 모델 실행 스크립트를 다운 받기 위해 EC2에 git을 설치합니다. 

```bash
sudo yum install git -y
```
- GID 저장소를 clone합니다. 
```bash
git clone https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day.git
cd ./FSI-Gameday-General-Immersion-Day/
chmod u+x ./*.sh
```

- Model API는 Tensorflow FLask 컨테이너로 이미 생성되어 있습니다. 
- EC2에 docker를 설치합니다.   
```angular2html
./install_back.sh
```

- 모두 설치되었다면 ec2-user 계정에 docker 실행 권한이 변경되어 기존 터미널 세션을 닫고, 다시 터미널에 진입합니다.
 
### 3. Model API 기동  
- Model API 기동 스크립트를 실행합니다. 
```bash
./get_ready.sh
```

 - Tensorflow에 Transformer Bet모델을 로드한 다음, Flask Rest API서버가 기동됩니다. 로그를 확인한 뒤 APU를 브라우저에서 호출해봅니다.

```bash
http://{퍼블릭 IPv4 주소}:8080/

# example: 
http://3.34.140.0:8080/
```

### 4. Model API TEST
- 초록색 TAG layer를 클릭하면 입력창이 열립니다. 
- 한글 문장을 입력하고 서버에 요청하면 감정 분류 정보를 응답하는 것을 확인합니다. 
![screenshot9](https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day/blob/main/pic/step_8.png?raw=true)
![screenshot10](https://github.com/hyeonsangjeon/FSI-Gameday-General-Immersion-Day/blob/main/pic/step_77.png?raw=true)
