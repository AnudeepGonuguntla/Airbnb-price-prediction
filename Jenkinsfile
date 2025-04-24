pipeline {
    agent any
    environment {
        DOCKER_IMAGE = "anudeep16/airbnb-streamlit:v${env.BUILD_NUMBER}"
    }
    stages {
        stage('Clone Repository') {
            steps {
                git branch: '${BRANCH_NAME}', 
                    url: 'https://github.com/AnudeepGonuguntla/Airbnb-price-prediction.git',
                    credentialsId: 'github-credentials'
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    dockerImage = docker.build("${DOCKER_IMAGE}")
                }
            }
        }
        stage('Push Docker Image') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    script {
                        sh 'echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin'
                        sh "docker push ${DOCKER_IMAGE}"
                    }
                }
            }
        }
        stage('Deploy') {
            when {
                branch 'master'
            }
            steps {
                script {
                    sh 'docker stop airbnb-app-container || true'
                    sh 'docker rm airbnb-app-container || true'
                    sh "docker run -d -p 8503:8501 --name airbnb-app-container ${DOCKER_IMAGE}"
                }
            }
        }
    }
    post {
        always {
            sh 'docker system prune -f'
            echo 'Pipeline finished.'
        }
    }
}
