pipeline {
    agent any
    parameters {
        string(name: 'HOST_PORT', defaultValue: '8503', description: 'Host port to map to container port 8501')
    }
    environment {
        IMAGE_NAME = 'anudeep16/airbnb-streamlit'
        IMAGE_TAG = "v${env.BUILD_NUMBER}"
        CONTAINER_NAME = 'airbnb-app-container'
    }
    stages {
        stage('Clean Workspace') {
            steps {
                deleteDir()
            }
        }

        stage('Clone Repo') {
            steps {
                git branch: 'main', 
                    url: 'https://github.com/AnudeepGonuguntla/Airbnb-price-prediction.git',
                    credentialsId: 'github-credentials'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "Building Docker image without cache..."
                bat "docker build --no-cache -t %IMAGE_NAME%:%IMAGE_TAG% ."
            }
        }

        stage('Push Docker Image') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    bat 'echo %DOCKER_PASS% | docker login -u %DOCKER_USER% --password-stdin'
                    bat "docker push %IMAGE_NAME%:%IMAGE_TAG%"
                }
            }
        }

        stage('Cleanup Existing Container') {
            steps {
                bat """
                for /f %%i in ('docker ps -a -q --filter "name=%CONTAINER_NAME%"') do (
                    docker stop %%i
                    docker rm %%i
                )
                """
            }
        }

        stage('Run New Container') {
            steps {
                bat "docker run -d -p %HOST_PORT%:8501 --name %CONTAINER_NAME% %IMAGE_NAME%:%IMAGE_TAG%"
            }
        }

        stage('Check Container Logs (Optional Debug)') {
            steps {
                echo 'Checking container logs (this is optional, remove if not needed)'
                bat "docker logs %CONTAINER_NAME%"
            }
        }
    }

    post {
        always {
            bat "docker system prune -f"
            echo 'Pipeline finished.'
        }
    }
}
