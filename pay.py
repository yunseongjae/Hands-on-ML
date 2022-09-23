import streamlit as st
import pandas as pd

class check:
    def processing_survey(file_name):
        df = pd.read_excel(file_name)
        df = df[['1. 이름을 적어주세요.', '2. 핸드폰 번호를 기입해주세요.']]
        df.columns = ['이름', '번호']
        df_copy = df.copy()
        df['번호'] = df['번호'].apply(lambda x: str(x)[-4:])
        df['key'] = df['이름'] + df['번호']
        return df, df_copy

    def processing_payple(file_name):
        df = pd.read_excel(file_name, header = 1)
        choice = st.radio("챌린지명을 골라주세요", df['상품명'].unique())
        df = df[df['상품명'] == choice]
        df = df[['구매자', '휴대폰번호', '상태', '거래일자', '거래시간']]
        df = df.drop_duplicates(subset = ['구매자', '휴대폰번호'],keep='last')
        df = df[['구매자', '휴대폰번호', '상태']]
        df.columns = ['이름', '번호','상태']
        df['번호'] = df['번호'].apply(lambda x: str(x)[-4:])
        df['key'] = df['이름'] + df['번호']
        return df


st.header("결제 확인 시스템")

uploaded_file1 = st.file_uploader("설문지 파일을 선택해주세요")

if uploaded_file1 is not None:
    a, a_copy = check.processing_survey(uploaded_file1)

uploaded_file2 = st.file_uploader("페이플 파일을 선택해주세요")
if uploaded_file2 is not None:
    b = check.processing_payple(uploaded_file2)

if uploaded_file1 is not None and uploaded_file2 is not None:
    df_total = pd.merge(left =a, right  = b, on = 'key', how = 'left')
    df_total['결제여부'] = ~(df_total['이름_y'].isnull())
    df_result = df_total[df_total['결제여부'] == True][['이름_x', '번호_x','상태']]
    df_result.columns = ['이름', '번호','상태']
    df_total_result = pd.merge(
    left = a_copy, right = df_result,
    left_on = a_copy['번호'].apply(lambda x: str(x)[-4:]), right_on = '번호', how = 'left')
    df_total_result['번호_x'] = df_total_result['번호_x'].apply(lambda x: str(x))
    df_total_result = df_total_result[['이름_x', '번호_x', '이름_y','상태']]
    df_total_result['이름_y'] = (~df_total_result['이름_y'].isnull())
    df_total_result.columns = ['이름', '번호', '결제여부','상태']
    df_total_result.drop_duplicates(subset = ['이름', '번호','상태'], inplace = True)
    df_total_result['상태'] = df_total_result['상태'].replace('성공','결제').replace('취소','환불').fillna('미결제')
    st.subheader('전체 명단입니다.')
    st.dataframe(df_total_result)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('결제 명단')
        st.dataframe(df_total_result[df_total_result['상태'] == '결제'])
    with col2:
        st.subheader('미결제 명단')
        st.dataframe(df_total_result[df_total_result['상태'] == '미결제'])
# st.table(df_total[df_total['결제여부'] == False][['이름_x', '번호_x']])
    # a = check().processing_survey(uploaded_file.name)
    # st.dataframe(a)
