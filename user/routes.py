# import app
# from user.models import User
#
# @app.route('/user/signup', methods=['POST'])
# def signup():
#   return User().signup()
#
# @app.route('/user/signout')
# def signout():
#   return User().signout()
#
# @app.route('/user/login', methods=['POST'])
# def login():
#   return User().login()


from flask import Blueprint, render_template

# Create a Blueprint object to define routes
user_bp = Blueprint('user', __name__)

@user_bp.route('/user/signup', methods=['POST'])
def signup():
    # Your signup logic here
    return render_template('.html')
