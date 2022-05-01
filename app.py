from run_pplm import run_pplm_example
import streamlit as st

st.set_page_config(layout="wide")

if __name__ == "__main__":
    print()
    padding_top = 0
    st.markdown(f"""
        <style>
            .reportview-container .main .block-container{{
                padding-top: 0 rem;
            }}
        </style>""",
                unsafe_allow_html=True)
    st.header("Controlled Text Generation by using Plug and Play Language Models")

    with st.sidebar:
        st.markdown("#### PPLM Model & decoder settings")
        bow = st.multiselect(
            'Bag-of-words',
            (
            "🖥️ computers", "🧚 fantasy", "🔪 kitchen", "‍💼 legal", "🎖 military", "🤴 politics", "😊 positive_words",
            "🕌 religion", "🧪 science",
            "🚀 space", "🚗 automotive", "🏐 sports", "🍲 food", "🧳 business", "💰 money", "🌨️ winter",
            "👮 occupation", "🏖️ holidays", "🗳️ elections"))

        discriminator = st.radio("Discriminators",
                                 ("🐭 clickbait", "🙏 non clickbait", "🙂 positive sentiment", "🙁 neg sentiment"))
        step_size = st.slider('Step size', 0.01, 0.1, (0.03))
        num_samples = st.slider('Num Samples', 1, 30, (3))
        window_length = st.slider('Window length', 5, 25, (5))
        num_iterations = st.slider('Num iterations (impacts gen. time)', 1, 30, (3))
        gen_len = st.slider('Gen. length (impacts gen. time)', 5, 80, (30))
        kl_scale = st.slider('KL-scale', 0.0, 0.99, (0.01))
        gm_scale = st.slider('GM-scale', 0.0, 0.99, (0.95))
        gamma = st.slider('gamma', 0.0, 10.0, 1.5)
        use_sampling = st.checkbox('Use sampling', value=True)

        params = {'bow': ";".join([word[2:].strip() for word in bow]), 'discriminator': discriminator[2:].strip(), 'step_size': step_size,
                  'num_samples': num_samples, 'window_length': window_length, 'num_iterations': num_iterations,
                  'gen_len': gen_len, 'kl_scale': kl_scale, 'gm_scale': gm_scale, 'gamma': gamma,
                  'use_sampling': use_sampling}

    # st.write(params)
    cond_text = st.text_input('Please enter the conditional text')
    if st.button("Run PPLM: Generate Text"):
        st.write(cond_text)
        run_pplm_example(
            cond_text=cond_text,
            discrim=params['discriminator'],
            num_samples=params['num_samples'],
            bag_of_words=params['bow'],
            length=params['gen_len'],
            stepsize=params['step_size'],
            sample=params['use_sampling'],
            num_iterations=params['num_iterations'],
            window_length=params['window_length'],
            gamma=params['gamma'],
            gm_scale=params['gm_scale'],
            kl_scale=params['kl_scale'],
            verbosity='regular'
        )
    # run_pplm_example(
    #     cond_text="The moment",
    #     num_samples=3,
    #     bag_of_words='automotive',
    #     length=50,
    #     stepsize=0.03,
    #     sample=True,
    #     num_iterations=3,
    #     window_length=5,
    #     gamma=1.5,
    #     gm_scale=0.95,
    #     kl_scale=0.01,
    #     verbosity='regular'
    # )
