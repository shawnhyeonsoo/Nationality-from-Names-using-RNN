# Guessing the nationality of a person from his or her name
A program implemented with RNN, guessing nationality of a person given his or her name as the input
</br>
</br>
Every country has specific characteristics of names given to a person; names such as 'Wang' or 'Lee' are normally
given to people from an Asian background. 
</br>
</br>
Although not a strict rule, it is common around the globe. For example, according to telegraph: the most popular name
for a boy across US and Canada was 'Liam', and that for a girl was 'Emma' in 2018. (https://www.telegraph.co.uk/travel/lists/most-popular-names-around-the-world-what-they-mean/)
</br>
</br>
As such, certain trends do exist in different countries for names given to people. </br>
This program, although not a novel idea itself, was made as a meaningful exercise while studying ML techniques and Natural
Language Processing theories.
</br>
</br>
The training data was prepared beforehand, and is implemented so that the program locally finds the data to train itself.
The code is based on character-level Recurrent Neural Network: </br>
It transforms the data given in numerous .txt files into tensor format and operates calculations in order to train from
the data.
</br>
At the last stage, the program takes input from the command line, and returns three (can be adjusted from the code itself)
most probable nationality of the name.
</br>
</br>
</br>
# 주어진 이름을 바탕으로 국적을 예측하는 프로그램

세계 각국에서 사용하는 언어가 다른만큼 각 사람이 갖는 이름도 다양한데, 이름이 주어졌을 때 해당 사람의 국적을 예측하는 프로그램이다. </br>
나라마다 이름을 짓는 규칙이 어느정도 있기 마련인데, 예를 들어 우리나라에서 사용하는 '철수'라는 이름은 다른 나라에서 잘 사용하진 않듯, 미국에서 사용하는 'Peter'
와 같은 이름을 우리나라 사람에게 잘 지어주진 않는다. </br>
</br>
이러한 어느정도의 규칙은 세계 각국에 있는데, 예를 들어 텔레그래프에 따르면 2018년 미국과 캐나다 전지역에서 가장 많이 사용된 남자 이름은 'Liam'이었으며, 가장 많이
사용된 여자 이름은 'Emma'였다고 한다. (https://www.telegraph.co.uk/travel/lists/most-popular-names-around-the-world-what-they-mean/)
</br>
</br>
본 프로그램은 새로운 알고리즘이나 아이디어는 아니지만, 필자가 ML과 자연어처리를 공부하는 과정에 수행한 프로젝트라는 점에 의미를 두었다.
</br>
프로그램을 위한 학습 데이터는 사전에 준비되어 있었으며, 프로그램은 해당 데이터를 로컬 디렉토리에서 찾아 올바른 형태로 변환시킨 후, 연산을 진행하는 과정으로
설계되었다. 학습 데이터는 .txt 확장자로 되어있으며, 커맨드창에 입력될 입력값에 대한 프로그램의 출력은 해당 이름에 대해 올바른 국적일 확률이 가장 높은 3개(수정 가능)
의 나라이다. </br>
전체적으로 문자 단위 RNN을 바탕으로 짜여졌으며, pytorch를 주로 이용해 설계되었다. 
