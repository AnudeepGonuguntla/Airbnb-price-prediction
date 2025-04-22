pipeline {
    agent any
    environment {
        DOCKER_IMAGE = "anudeepgonuguntla/airbnb-streamlit:latest"
        DOCKER_CREDENTIALS_ID = "docker-hub-credentials"
    }
    stages {
        stage('Checkout') {
            steps {
                script {
                    try {
                        git branch: 'main', url: 'https://github.com/AnudeepGonuguntla/Airbnb-price-prediction.git'
                        echo "Successfully checked out repository"
                    } catch (Exception e) {
                        error "Failed to checkout repository: ${e.message}"
                    }
                }
            }
        }
        stage('Verify Docker') {
            steps {
                script {
                    try {
                        bat 'docker --version'
                        bat 'docker info'
                        echo "Docker is accessible"
                    } catch (Exception e) {
                        error "Docker verification failed: ${e.message}"
                    }
                }
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    try {
                        def image = docker.build("${DOCKER_IMAGE}")
                        echo "Successfully built Docker image: ${DOCKER_IMAGE}"
                    } catch (Exception e) {
                        error "Failed to build Docker image: ${e.message}"
                    }
                }
            }
        }
        stage('Test Docker Image') {
            steps {
                script {
                    try {
                        def container = docker.image("${DOCKER_IMAGE}").run('-p 8502:8501 --name test-container')
                        sleep 20 // Increased to ensure app starts
                        container.stop()
                        echo "Docker image test passed"
                    } catch (Exception e) {
                        error "Docker image test failed: ${e.message}"
                    }
                }
            }
        }
        stage('Push to Docker Hub') {
            steps {
                script {
                    try {
                        docker.withRegistry('https://index.docker.io/v1/', "${DOCKER_CREDENTIALS_ID}") {
                            docker.image("${DOCKER_IMAGE}").push()
                        }
                        echo "Successfully pushed Docker image to Docker Hub"
                    } catch (Exception e) {
                        error "Failed to push Docker image: ${e.message}"
                    }
                }
            }
        }
    }
    post {
        always {
            script {
                // Use bat for Windows compatibility
                bat 'docker rm -f test-container || exit 0'
            }
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs for details.'
        }
    }
}
