import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# 加载原始数据
def load_public_data():
    df_public = pd.read_csv("Reviews.csv", encoding="utf-8-sig")
    df_public = df_public.head(10000)
    print(" 原始数据加载完成，数据量：", len(df_public))
    return df_public

df_original = load_public_data()


# EDA
def explore_data(df):
    print("EDA开始")
    print("数据前5行：")
    print(df.head())

    print("\n数据结构信息：")
    print(df.info())

    print("\nscore统计描述：")
    print(df["Score"].describe())

    print("\n缺失值统计：")
    missing_values = df.isnull().sum()
    print(missing_values)

    print("\n重复值统计：")
    total_duplicates = df.duplicated().sum()
    text_duplicates = df.duplicated(subset=["Text"]).sum()
    print(f"   全表重复行数：{total_duplicates}")
    print(f"   评论文本重复行数：{text_duplicates}")

    print("\n评分分布：")
    score_distribution = df["Score"].value_counts().sort_index()
    print(score_distribution)
    print("EDA结束")
    return missing_values, text_duplicates

missing_vals, text_dups = explore_data(df_original)


# 数据清洗
def clean_data(df):
    print("\n数据清洗开始")
    #去重
    df_cleaned = df.drop_duplicates(subset=["Text"], keep="first", inplace=False)
    print(f"   去重后数据行数：{len(df_cleaned)}（原数据：{len(df)}）")

    #处理缺失值：去掉核心、非核心添0
    df_cleaned = df_cleaned.dropna(subset=["Text", "Score"], inplace=False)
    df_cleaned["Id"] = df_cleaned["Id"].fillna(0).astype(int)
    print(f"   处理缺失值后数据行数：{len(df_cleaned)}")

    #剔除无效文本
    def is_valid_text(text):
        if pd.isna(text):
            return False
        text = str(text).strip()
        return len(text) > 5

    df_cleaned = df_cleaned[df_cleaned["Text"].apply(is_valid_text)]
    print(f"   剔除无效文本后数据行数：{len(df_cleaned)}")

    #文本去噪
    def clean_special_chars(text):
        text = str(text).strip()
        # 保留英文、数字和空格、去除多余空格
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # 新增清洗后文本列
    df_cleaned["clean_text"] = df_cleaned["Text"].apply(clean_special_chars)
    print("   文本去噪完成，新增列：clean_text")
    print("数据清洗结束")
    return df_cleaned

df_cleaned = clean_data(df_original)


# 预处理+打标签
#nltk.download("stopwords")

def preprocess_eng_and_build_label(df):
    print("\n文本预处理+打标签开始")
    eng_stop_words = set(stopwords.words("english"))
    print(f"   已加载stopwords，共包含 {len(eng_stop_words)} 个")

    #文本预处理
    def clean_text(text):
        text = str(text).strip().lower()
        word_list = text.split()
        filtered_word_list = [word for word in word_list if word not in eng_stop_words]
        cleaned_text = " ".join(filtered_word_list)

        return cleaned_text

    #批量预处理
    df_final = df.copy()
    df_final["final_text"] = df_final["clean_text"].apply(clean_text)
    print("   文本预处理完成，存放在final_text中")

    #打标签
    def build_sentiment_label(score):
        if score >= 4:
            return 1
        elif score == 3:
            return 0
        else:
            return -1

    df_final["sentiment"] = df_final["Score"].apply(build_sentiment_label)
    print("   标签构建完成，存放在sentiment中")

    #查看结果
    sentiment_dist = df_final["sentiment"].value_counts().rename({1: "Positive", 0: "Neutral", -1: "Negative"})
    print("\n   标签分布：")
    print(sentiment_dist)
    print("预处理+打标签结束")
    return df_final

df_final = preprocess_eng_and_build_label(df_cleaned)


#保存
def save_final_result(df, original_df):
    print("\n数据集保存：")
    core_columns = ["Id", "Text", "final_text", "Score", "sentiment"]
    df_final_output = df[core_columns].copy()

    #df_final_output.to_csv("cleaned_ecommerce_reviews.csv", index=False, encoding="utf-8-sig")
    #df_final_output.to_excel("cleaned_ecommerce_reviews.xlsx", index=False, engine="openpyxl")
    print("   CSV：cleaned_ecommerce_reviews.csv")
    print("   Excel：cleaned_ecommerce_reviews.xlsx")

    #计算
    total_original = len(original_df)
    total_final = len(df_final_output)
    data_quality_rate = (total_final / total_original) * 100
    positive_ratio = (df_final_output["sentiment"] == 1).sum() / total_final * 100

    print("\n最终成果：")
    print(f"处理数据 {total_original} 条，输出 {total_final} 条")
    print(f"有效率提升至 {data_quality_rate:.2f}%")
    print(f"正面评论占比 {positive_ratio:.2f}%")

save_final_result(df_final, df_original)