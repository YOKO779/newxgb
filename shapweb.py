import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import shap
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# 设置 matplotlib 显示中文字体
rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体 SimSun
rcParams['axes.unicode_minus'] = False      # 解决负号 '-' 显示为方块的问题

# 页面配置
st.set_page_config(page_title='老年糖尿病患者衰弱风险预测')

def main():
    # 检查模型路径
    model_path = 'xgb_model.pkl'
    if not os.path.exists(model_path):
        st.error("模型文件不存在，请检查路径！")
        return
    lgbm = joblib.load(model_path)  # 加载模型

    # 定义 Subject 类
    class Subject:
        def __init__(self, 认知障碍, 体育活动量, 慢性疼痛, 营养状态, 肌少症, 查尔斯共病指数, 糖尿病肾病):
            self.data = {
                "Cognition impaired": [认知障碍],
                "Physical activity": [体育活动量],
                "Chronic pain": [慢性疼痛],
                "Nutritional status": [营养状态],
                "Sarcopenia": [肌少症],
                "Charlson Comorbidity Index": [查尔斯共病指数],
                "Diabetic nephropathy": [糖尿病肾病]
            }

        def make_predict(self, lgbm):
            # 构造输入数据
            df_subject = pd.DataFrame(self.data)

            # 按照模型训练时的特征顺序重新排列列顺序
            expected_columns = [
                'Cognition impaired',
                'Physical activity',
                'Chronic pain',
                'Nutritional status',
                'Diabetic nephropathy',
                'Charlson Comorbidity Index',
                'Sarcopenia'
            ]
            df_subject = df_subject[expected_columns]

            # 模型预测
            prediction = lgbm.predict_proba(df_subject)[:, 1]
            adjusted_prediction = np.round(prediction * 100, 2)
            st.write(f"""
                <div class='all'>
                    <p style='text-align: center; font-size: 20px;'>
                        <b>模型预测老年糖尿病患者衰弱风险为 {adjusted_prediction[0]} %</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # SHAP 可视化
            explainer = shap.TreeExplainer(lgbm)
            shap_values = explainer.shap_values(df_subject)

            # 检查是否为多分类情况
            if isinstance(explainer.expected_value, list):
                expected_value = explainer.expected_value[1]
                shap_value = shap_values[1][0, :]
            else:
                expected_value = explainer.expected_value
                shap_value = shap_values[0, :]

            # 绘制 SHAP 力图
            shap.force_plot(
                expected_value,    # 基准值
                shap_value,        # 特征贡献值
                df_subject.iloc[0, :],  # 当前样本特征
                matplotlib=True
            )
            st.pyplot(plt.gcf())
            plt.clf()  # 清除图像，防止残留

    # 英文列名到中文显示的映射（仅用于前端显示）
    column_name_mapping = {
        "Cognition impaired": "认知障碍",
        "Physical activity": "体育活动量",
        "Chronic pain": "慢性疼痛",
        "Nutritional status": "营养状态",
        "Sarcopenia": "肌少症",
        "Charlson Comorbidity Index": "查尔斯共病指数",
        "Diabetic nephropathy": "糖尿病肾病"
    }

    # UI 配置
    st.markdown("""
                <div class='all'>
                    <h1 style='text-align: center;'>老年糖尿病患者衰弱风险预测</h1>
                </div>
                """, unsafe_allow_html=True)

    # 前端显示中文，后台仍用英文构造数据
    认知障碍 = st.selectbox(column_name_mapping["Cognition impaired"] + " (是 = 1, 否 = 0)", [1, 0], index=1)
    体育活动量 = st.selectbox(column_name_mapping["Physical activity"] + " (低活动量 = 1, 中活动量 = 2, 高活动量 = 3)", [1, 2, 3], index=0)
    慢性疼痛 = st.selectbox(column_name_mapping["Chronic pain"] + " (有 = 1, 无 = 0)", [1, 0], index=1)
    营养状态 = st.selectbox(column_name_mapping["Nutritional status"] + " (营养良好 = 0, 营养不良风险 = 1, 营养不良 = 2)", [0, 1, 2], index=1)
    肌少症 = st.selectbox(column_name_mapping["Sarcopenia"] + " (是 = 1, 否 = 0)", [1, 0], index=1)
    查尔斯共病指数 = st.number_input(column_name_mapping["Charlson Comorbidity Index"], value=2, min_value=0, max_value=30, step=1)
    糖尿病肾病 = st.selectbox(column_name_mapping["Diabetic nephropathy"] + " (有 = 1, 无 = 0)", [1, 0], index=1)

    if st.button(label="提交"):
        user = Subject(认知障碍, 体育活动量, 慢性疼痛, 营养状态, 肌少症, 查尔斯共病指数, 糖尿病肾病)
        user.make_predict(lgbm)

if __name__ == "__main__":
    main()
