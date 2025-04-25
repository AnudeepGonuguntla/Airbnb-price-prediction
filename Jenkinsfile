pipeline {
    agent any

    environment {
        IMAGE_NAME = "anudeep16/airbnb-streamlit"
        IMAGE_TAG = "v1"
        CONTAINER_NAME = "airbnb-app-container"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Image') {
            steps {
                echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
                bat "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
            }
        }

        stage('Stop and Remove Existing Container') {
            steps {
                script {
                    echo "Stopping and removing existing container if it exists..."
                    try {
                        bat """
                        for /f %%i in ('docker ps -a -q --filter "name=${CONTAINER_NAME}"') do (
                            docker stop %%i
                            docker rm %%i
                        )
                        """
                    } catch (Exception e) {
                        echo "No running container to stop or error during cleanup: ${e.getMessage()}"
                    }
                }
            }
        }

        stage('Run New Container') {
            steps {
                echo "Running new container: ${CONTAINER_NAME}"
                bat """
                docker run -d -p 8501:8501 --name ${CONTAINER_NAME} ${IMAGE_NAME}:${IMAGE_TAG}
                """
            }
        }
    }

    post {
        always {
            echo "Cleaning up unused Docker resources..."
            bat "docker system prune -f -a"
        }
    }
}
