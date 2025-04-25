pipeline {
    agent any

    parameters {
        string(name: 'HOST_PORT', defaultValue: '8503', description: 'Host port to map to container port 8501')
    }

    environment {
        IMAGE_NAME = 'anudeep16/airbnb-streamlit'
        IMAGE_TAG = "v${env.BUILD_NUMBER}"         // Correct way to build tag with build number
        CONTAINER_NAME = 'airbnb-app-container'
    }

    stages {
        stage('Clone Repo') {
            steps {
                git branch: 'main', 
                    url: 'https://github.com/AnudeepGonuguntla/Airbnb-price-prediction.git',
                    credentialsId: 'github-credentials'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "Building Docker image using cache..."
                bat "docker build -t ${env.IMAGE_NAME}:${env.IMAGE_TAG} ."
            }
        }

        stage('Push Docker Image') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    bat 'echo %DOCKER_PASS% | docker login -u %DOCKER_USER% --password-stdin'
                    bat "docker push ${env.IMAGE_NAME}:${env.IMAGE_TAG}"
                }
            }
        }

        stage('Cleanup Existing Container') {
            when {
                branch 'main'
            }
            steps {
                bat """
                FOR /F %%i IN ('docker ps -a -q --filter "name=${env.CONTAINER_NAME}"') DO (
                    docker stop %%i
                    docker rm %%i
                )
                """
            }
        }

        stage('Run New Container') {
            when {
                branch 'main'
            }
            steps {
                bat "docker run -d -p ${params.HOST_PORT}:8501 --name ${env.CONTAINER_NAME} ${env.IMAGE_NAME}:${env.IMAGE_TAG}"
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
