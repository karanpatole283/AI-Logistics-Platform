# AI-Logistics-Platform
objective is to revolutionize air logistics by leveraging advanced technologies such as AIML, to provide real-time insights, optimize resource usage, and streamline supply chain operations

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Authentication</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #3b82f6, #9333ea);
        }
    </style>
</head>
<body class="gradient-bg min-h-screen flex items-center justify-center">
    <div class="bg-white shadow-lg rounded-lg p-8 w-full max-w-md">
        <!-- Toggle between forms -->
        <div id="form-toggle" class="text-center mb-6">
            <button id="show-login" class="text-blue-600 font-semibold mr-4">Login</button>
            <button id="show-signup" class="text-gray-400">Sign Up</button>
            <button id="show-reset" class="text-gray-400 ml-4">Forgot Password?</button>
        </div>

        <!-- Login Form -->
        <div id="login-form">
            <h2 class="text-xl font-bold text-gray-800 mb-4">Login</h2>
            <form id="login" class="space-y-4">
                <input type="text" id="login-username" placeholder="Username" class="w-full px-4 py-2 border rounded">
                <input type="password" id="login-password" placeholder="Password" class="w-full px-4 py-2 border rounded">
                <button type="submit" class="w-full bg-blue-600 text-white py-2 rounded">Login</button>
            </form>
        </div>

        <!-- Signup Form -->
        <div id="signup-form" class="hidden">
            <h2 class="text-xl font-bold text-gray-800 mb-4">Sign Up</h2>
            <form id="signup" class="space-y-4">
                <input type="text" id="signup-username" placeholder="Username" class="w-full px-4 py-2 border rounded">
                <input type="email" id="signup-email" placeholder="Email" class="w-full px-4 py-2 border rounded">
                <input type="password" id="signup-password" placeholder="Password" class="w-full px-4 py-2 border rounded">
                <button type="submit" class="w-full bg-purple-600 text-white py-2 rounded">Sign Up</button>
            </form>
        </div>

        <!-- Password Reset Form -->
        <div id="reset-form" class="hidden">
            <h2 class="text-xl font-bold text-gray-800 mb-4">Reset Password</h2>
            <form id="reset-request" class="space-y-4">
                <input type="email" id="reset-email" placeholder="Enter your email" class="w-full px-4 py-2 border rounded">
                <button type="submit" class="w-full bg-yellow-500 text-white py-2 rounded">Send Reset Link</button>
            </form>
            <form id="reset-password" class="space-y-4 hidden">
                <input type="text" id="reset-token" placeholder="Enter your reset token" class="w-full px-4 py-2 border rounded">
                <input type="password" id="new-password" placeholder="New Password" class="w-full px-4 py-2 border rounded">
                <button type="submit" class="w-full bg-green-600 text-white py-2 rounded">Reset Password</button>
            </form>
        </div>

        <p class="text-center text-sm text-gray-500 mt-6">© 2024 Your App. All rights reserved.</p>
    </div>

    <script>
        // Form Toggle
        const showLogin = document.getElementById('show-login');
        const showSignup = document.getElementById('show-signup');
        const showReset = document.getElementById('show-reset');

        const loginForm = document.getElementById('login-form');
        const signupForm = document.getElementById('signup-form');
        const resetForm = document.getElementById('reset-form');

        showLogin.addEventListener('click', () => {
            resetForm.classList.add('hidden');
            signupForm.classList.add('hidden');
            loginForm.classList.remove('hidden');
            showLogin.classList.add('text-blue-600', 'font-semibold');
            showSignup.classList.remove('text-blue-600', 'font-semibold');
            showReset.classList.remove('text-blue-600', 'font-semibold');
        });

        showSignup.addEventListener('click', () => {
            loginForm.classList.add('hidden');
            resetForm.classList.add('hidden');
            signupForm.classList.remove('hidden');
            showSignup.classList.add('text-blue-600', 'font-semibold');
            showLogin.classList.remove('text-blue-600', 'font-semibold');
            showReset.classList.remove('text-blue-600', 'font-semibold');
        });

        showReset.addEventListener('click', () => {
            loginForm.classList.add('hidden');
            signupForm.classList.add('hidden');
            resetForm.classList.remove('hidden');
            showReset.classList.add('text-blue-600', 'font-semibold');
            showLogin.classList.remove('text-blue-600', 'font-semibold');
            showSignup.classList.remove('text-blue-600', 'font-semibold');
        });

        // Backend Integration (Add fetch calls to backend endpoints)
        document.getElementById('login').addEventListener('submit', async (event) => {
            event.preventDefault();
            const username = document.getElementById('login-username').value;
            const password = document.getElementById('login-password').value;

            const response = await fetch('/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            const data = await response.json();
            if (response.ok) {
                alert("Login Successful");
                console.log(data); // Handle JWT token
            } else {
                alert(data.detail || "Login failed");
            }
        });

        document.getElementById('signup').addEventListener('submit', async (event) => {
            event.preventDefault();
            const username = document.getElementById('signup-username').value;
            const email = document.getElementById('signup-email').value;
            const password = document.getElementById('signup-password').value;

            const response = await fetch('/auth/signup', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, email, password })
            });
            const data = await response.json();
            if (response.ok) {
                alert("Sign up Successful");
            } else {
                alert(data.detail || "Sign up failed");
            }
        });

        document.getElementById('reset-request').addEventListener('submit', async (event) => {
            event.preventDefault();
            const email = document.getElementById('reset-email').value;

            const response = await fetch('/auth/request-password-reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email })
            });
            const data = await response.json();
            if (response.ok) {
                alert("Password reset link sent");
            } else {
                alert(data.detail || "Failed to send reset link");
            }
        });

        document.getElementById('reset-password').addEventListener('submit', async (event) => {
            event.preventDefault();
            const token = document.getElementById('reset-token').value;
            const newPassword = document.getElementById('new-password').value;

            const response = await fetch('/auth/reset-password', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ token, new_password: newPassword })
            });
            const data = await response.json();
            if (response.ok) {
                alert("Password reset successful");
            } else {
                alert(data.detail || "Password reset failed");
            }
        });

        document.getElementById('login').addEventListener('submit', (event) => {
    event.preventDefault();
    console.log("Redirecting to index.html");
    window.location.href = 'i1.html';
});

    </script>
</body>
</html>
