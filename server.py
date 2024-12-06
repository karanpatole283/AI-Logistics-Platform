import http.server
import socketserver
from os import path
from flask import Flask, request, render_template, redirect, url_for

my_host_name = 'localhost'
my_port = 8888
my_html_folder_path = 'C:\team axeron\frontend\public'
my_home_page_file_path = 'index.html'

# Existing HTTP server class
class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', path.getsize(self.getPath()))
        self.end_headers()

    def getPath(self):
        if self.path == '/':
            content_path = path.join(
                my_html_folder_path, my_home_page_file_path)
        else:
            content_path = path.join(my_html_folder_path, str(self.path).split('?')[0][1:])
        return content_path

    def getContent(self, content_path):
        with open(content_path, mode='r', encoding='utf-8') as f:
            content = f.read()
        return bytes(content, 'utf-8')

    def do_GET(self):
        self._set_headers()
        self.wfile.write(self.getContent(self.getPath()))

# Flask app for handling login functionality
app = Flask(__name__)

# Sample data for user authentication
users = {
    "user1": "password1",
    "user2": "password2"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            return redirect(url_for('index'))  # Redirect to the main page on successful login
        else:
            return "Invalid credentials, please try again."
    return render_template('index.html')

# Run Flask app and HTTP server concurrently
if __name__ == '__main__':
    import threading

    # Function to run Flask app
    def run_flask_app():
        app.run(host=my_host_name, port=my_port + 1, debug=True)

    # Start Flask app in a separate thread
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    # Run the existing HTTP server
    my_handler = MyHttpRequestHandler
    with socketserver.TCPServer(("", my_port), my_handler) as httpd:
        print("Http Server Serving at port", my_port)
        httpd.serve_forever()
