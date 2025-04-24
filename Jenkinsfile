pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'anudeepgonuguntla/airbnb-streamlit:latest'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git 'https://github.com/AnudeepGonuguntla/Airbnb-price-prediction.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    dockerImage = docker.build("${env.DOCKER_IMAGE}")
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    script {
                        sh 'echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin'
                        sh "docker push ${env.DOCKER_IMAGE}"
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished.'
        }
    }
}
