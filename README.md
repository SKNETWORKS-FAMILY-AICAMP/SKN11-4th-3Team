# SKN11-4TH-3Team
- 주제 : LLM 기반 보드게임 룰 설명 & 맞춤형 추천 챗봇 구동 웹 어플리케이션 개발
- 개발기간 : 25.05.16~25.06.11
---
## 📑 Index

1. [팀 소개](#1-팀-소개)
2. [Overview](#2-overview)
3. [기술 스택](#3-기술-스택)
4. [시스템 아키텍처](#4-시스템-아키텍처)
5. [WBS](#5-wbs)
6. [요구사항 명세서](#6-요구사항-명세서)
7. [화면설계서](#7-화면설계서)
8. [테스트 계획 및 결과 보고서](#8-테스트-계획-및-결과-보고서)
9. [수행결과](#9-수행결과)
10. [한 줄 회고](#10-한-줄-회고)


## 1. 팀 소개
### 팀명 : BoardNavi
- Board + Navi의 합성어로, “보드게임 세계의 길잡이” 라는 뜻으로, 사용자가 보드게임이라는 낯선 세계에서 길을 잃지 않도록 규칙과 추천을 안내해주는 팀이라는 의미를 담고 있습니다.


### 👤 팀원
<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/238c9b42-e99f-4dad-acb1-c2a9a6720165" width="120" />
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/1c830dbd-d5f7-458c-91c2-5828b4f66a46" width="120" />
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/80891080-25e5-4a26-975e-21907c3e243f" width="120" />
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/752caeb5-d90d-4f93-b511-b1ec8a987ba0" width="120" />
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://github.com/Kimjeongwon12">김정원</a>
    </td>
    <td align="center">
      <a href="https://github.com/minjung2266">이민정</a>
    </td>
    <td align="center">
      <a href="https://github.com/Minor1862">정민호</a>
    </td>
    <td align="center">
      <a href="https://github.com/junoaplus">황준호</a>
    </td>
  </tr>
</table>
<br/>


## 2. Overview

  #### 📖 프로젝트 소개 
보드게임 봇 "🤖보비"는 보드게임 룰 설명과 추천 기능을 제공하는 LLM 기반 챗봇입니다. 챗봇은 사용자의 질문에 따라 게임 규칙을 설명하거나 취향에 맞는 게임을 추천해줍니다.

#### ⭐ 프로젝트 필요성
<table>
  <tr>
    <td>초보자들의 게임 선택 장애</td>
    <td>보드게임의 대중화로 다양한 게임이 출시되고 있지만, 초보 이용자들은 복잡한 룰을 이해하거나 자신의 취향에 맞는 게임을 고르는 데 어려움을 겪음</td>
  </tr>
  <tr>
    <td>보드게임 카페의 인력 문제</td>
    <td>보드게임 카페에서는 다양한 게임을 설명하고 추천할 수 있는 직원을 필요로 하지만, 폭넓은 게임 지식을 갖춘 인력을 채용하기란 쉽지 않음</td>
  </tr>
</table>

#### 🎯 프로젝트 목표

<table>
  <tr>
    <td>보드게임 룰 설명 챗봇 구현</td>
    <td>사용자의 질문에 정확하고 간결한 게임 규칙을 제공</td>
  </tr>
  <tr>
    <td>보드게임 추천 기능 제공</td>
    <td>게임 방법, 인원 수, 테마 등을 기반으로 유사도 분석을 통해 최적의 보드게임을 추천</td>
  </tr>
  <tr>
    <td>도메인 특화 지식 반영</td>
    <td>벡터DB 구축과 LLM 파인튜닝을 통해 보드게임에 특화된 지식 기반 챗봇 구축</td>
  </tr>
</table>

<hr>

## 3. 기술 스택

| 항목                | 내용 |
|---------------------|------|
| **Language**        | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| **Development**     | ![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)<br>![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)<br>![RunPod](https://img.shields.io/badge/RunPod-8A2BE2?style=for-the-badge)<br>![AWS](https://camo.githubusercontent.com/124e5f950a353173a8b04bd8f04ead73248482e0aeb9b7d7ad9330fd65cb665a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4157532532304543322d4646393930303f7374796c653d666f722d7468652d6261646765266c6f676f3d416d617a6f6e253230415753266c6f676f436f6c6f723d7768697465)     |
| **Embedding Model** | ![Hugging Face](https://img.shields.io/badge/HuggingFace-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black) |
| **Vector DB**       | ![FAISS](https://img.shields.io/badge/FAISS-009688?style=for-the-badge) |
| **LLM Model**       | ![gpt-3.5-turbo_fine_tune](https://img.shields.io/badge/gpt--3.5-4B91FF?style=for-the-badge&logo=openai&logoColor=white)<br> <img src="https://img.shields.io/badge/EXAONE-A50034?style=for-the-badge&logo=LG&logoColor=white"> |
|   **Framework**   | ![Django](https://camo.githubusercontent.com/4c4a57a11a83f99eafb6eaaaaf65ea43e0fc446fccbf8533aac7e9be1067aaf7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446a616e676f2d3039324532303f7374796c653d666f722d7468652d6261646765266c6f676f3d446a616e676f266c6f676f436f6c6f723d7768697465)                  |
| **Collaboration Tool** | ![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)


## 4. 시스템 구성도
![image](https://github.com/user-attachments/assets/bdb62f74-2dc7-4971-954e-99ccd5da33a4)



## 5. WBS
![image](https://github.com/user-attachments/assets/149f750d-fe87-44b8-8a67-7fc9278ac9ec)




## 6. 요구사항 명세서
![](image/requirements.png)


## 7. 화면설계서
![화면정의서 (1)-1](https://github.com/user-attachments/assets/031a81cb-cd1e-4320-96ba-e739ff0c5ff4)
![화면정의서 (1)-2](https://github.com/user-attachments/assets/ac1009ff-c2aa-4da7-9e26-275ffa4f932b)
![화면정의서 (1)-3](https://github.com/user-attachments/assets/7e27fd02-143d-48a6-9387-bd3aefb4a98b)
![화면정의서 (1)-4](https://github.com/user-attachments/assets/a0965bb6-2a89-4346-8230-e7c7ba938c34)
![화면정의서 (1)-5](https://github.com/user-attachments/assets/28b23524-0b2b-47d7-ba83-4211e2771ec9)
![화면정의서 (1)-6](https://github.com/user-attachments/assets/676c6b02-03c1-4ebc-b2d9-e7fab78d129c)
![화면정의서 (1)-7](https://github.com/user-attachments/assets/f43f4e10-98dc-4753-b24b-8d4eb0df386f)






## 8. 테스트 계획 및 결과 보고서

### 📖테스트 계획
![](image/test_preparing.png)

### 📖테스트 결과
- TEST01
![](image/result_test01.png)

- TEST02
![](image/result_test02.png)

- TEST03
![](image/result_test03.png)

- TEST04
![](image/result_test04.png)

- TEST05
  <table>
  <tr>
    <td>
      <img src="image/result_test05.png" alt="사진2" style="width: 100%; height: auto;">
    </td>
  </tr>
  </table>

## 9. 수행결과(테스트/시연 페이지)



## 10. 한 줄 회고                                                                                                               
>  김정원 :
>
> 이민정 : 
>
>  정민호 :                                                                  
>
>  황준호 : 

