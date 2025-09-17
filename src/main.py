from flask import render_template
from application.web import WebApp 

webapp = WebApp(__name__)

@webapp.get_engine().route("/")
@webapp.get_engine().route("/index")
def home_index():
    return render_template("index.html")

@webapp.get_engine().route("/about")
def home_about():
    return render_template("about.html")

webapp.start()