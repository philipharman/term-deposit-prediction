# mkdir -p ~/.streamlit/

# heroku ps:scale web=1

# echo "\
# [server]\n\
# port = $PORT\n\
# enableCORS = false\n\
# headless = true\n\
# \n\
# " > ~/.streamlit/config.toml
web: gunicorn bot:app
