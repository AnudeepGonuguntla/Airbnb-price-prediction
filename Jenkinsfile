pipeline {
    agent any

    environment {
        IMAGE_NAME = 'anudeep16/airbnb-streamlit'
        IMAGE_TAG = 'v1'
        CONTAINER_NAME = 'airbnb-app-container'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/AnudeepGonuguntla/Airbnb-price-prediction.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "Building Docker image: ${env.IMAGE_NAME}:${env.IMAGE_TAG}"
                bat "docker build -t ${env.IMAGE_NAME}:${env.IMAGE_TAG} ."
            }
        }

        stage('Stop and Remove Existing Container') {
            steps {
                echo "Stopping and removing existing container if it exists..."
                bat """
                FOR /F %%i IN ('docker ps -a -q --filter "name=${env.CONTAINER_NAME}"') DO (
                    docker stop %%i
                    docker rm %%i
                )
                """
            }
        }

        stage('Run New Container') {
            steps {
                echo "Running new container on port 8501..."
                bat "docker run -d -p 8501:8501 --name ${env.CONTAINER_NAME} ${env.IMAGE_NAME}:${env.IMAGE_TAG}"
            }
        }
    }

    post {
        always {
            echo 'Cleaning up unused Docker resources...'
            bat "docker system prune -f -a"
            echo '✅ Pipeline finished for Anudeep.'
        }
    }
}
