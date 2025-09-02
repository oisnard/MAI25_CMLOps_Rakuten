#!/bin/bash

# MLOps Kubernetes Deployment Script for dev-chris Branch
# This script deploys your complete MLOps stack to Kubernetes
# Usage: ./deploy.sh [deploy|delete|status|logs|port-forward]

set -e  # Exit on any error

# Configuration
NAMESPACE="dev-chris"
KUSTOMIZE_DIR="monitoring"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if we can connect to the cluster
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    print_success "kubectl is available and connected to cluster"
}

# Function to check if namespace exists
check_namespace() {
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        print_status "Namespace '$NAMESPACE' already exists"
    else
        print_status "Creating namespace '$NAMESPACE'"
        kubectl create namespace $NAMESPACE
        print_success "Namespace '$NAMESPACE' created"
    fi
}

# Function to deploy the stack
deploy() {
    print_status "Starting deployment of MLOps stack..."
    
    # Check prerequisites
    check_kubectl
    check_namespace
    
    # Change to the kustomize directory
    cd "$SCRIPT_DIR/$KUSTOMIZE_DIR"
    
    # Deploy using kustomize
    print_status "Deploying resources using kustomize..."
    kubectl apply -k .
    
    print_success "Deployment completed successfully!"
    
    # Show status
    print_status "Checking deployment status..."
    kubectl get all -n $NAMESPACE
    
    # Show services
    print_status "Services created:"
    kubectl get services -n $NAMESPACE
    
    print_success "MLOps stack is now deployed in namespace '$NAMESPACE'"
}

# Function to delete the stack
delete() {
    print_warning "This will delete all resources in namespace '$NAMESPACE'"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Deleting MLOps stack..."
        
        # Change to the kustomize directory
        cd "$SCRIPT_DIR/$KUSTOMIZE_DIR"
        
        # Delete using kustomize
        kubectl delete -k .
        
        # Delete namespace
        kubectl delete namespace $NAMESPACE
        
        print_success "MLOps stack deleted successfully!"
    else
        print_status "Deletion cancelled"
    fi
}

# Function to show status
status() {
    print_status "Checking status of MLOps stack..."
    
    # Check namespace
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        print_status "Namespace '$NAMESPACE' exists"
        
        # Show all resources
        kubectl get all -n $NAMESPACE
        
        # Show services
        kubectl get services -n $NAMESPACE
        
        # Show configmaps
        kubectl get configmaps -n $NAMESPACE
        
        # Show pod logs
        print_status "Recent pod events:"
        kubectl get events -n $NAMESPACE --sort-by='.lastTimestamp'
    else
        print_warning "Namespace '$NAMESPACE' does not exist"
    fi
}

# Function to show logs
logs() {
    print_status "Showing logs for MLOps stack..."
    
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        # Show logs for each deployment
        print_status "FastAPI logs:"
        kubectl logs -n $NAMESPACE -l app=fastapi-dev-chris --tail=50 || true
        
        print_status "Prometheus logs:"
        kubectl logs -n $NAMESPACE -l app=prometheus-dev-chris --tail=50 || true
        
        print_status "Grafana logs:"
        kubectl logs -n $NAMESPACE -l app=grafana-dev-chris --tail=50 || true
    else
        print_warning "Namespace '$NAMESPACE' does not exist"
    fi
}

# Function to set up port forwarding
port_forward() {
    print_status "Setting up port forwarding for MLOps stack..."
    
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        print_status "Port forwarding setup:"
        print_status "FastAPI: kubectl port-forward -n $NAMESPACE svc/fastapi-service-dev-chris 8000:8000"
        print_status "Prometheus: kubectl port-forward -n $NAMESPACE svc/prometheus-service-dev-chris 9090:9090"
        print_status "Grafana: kubectl port-forward -n $NAMESPACE svc/grafana-service-dev-chris 3000:3000"
        
        print_status "Starting port forwarding in background..."
        
        # Start port forwarding in background
        kubectl port-forward -n $NAMESPACE svc/fastapi-service-dev-chris 8000:8000 &
        kubectl port-forward -n $NAMESPACE svc/prometheus-service-dev-chris 9090:9090 &
        kubectl port-forward -n $NAMESPACE svc/grafana-service-dev-chris 3000:3000 &
        
        print_success "Port forwarding started!"
        print_status "Access your services at:"
        print_status "  FastAPI: http://localhost:8000"
        print_status "  Prometheus: http://localhost:9090"
        print_status "  Grafana: http://localhost:3000 (admin/admin)"
        
        print_warning "Press Ctrl+C to stop port forwarding"
        wait
    else
        print_warning "Namespace '$NAMESPACE' does not exist"
    fi
}

# Function to show help
show_help() {
    echo "MLOps Kubernetes Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  deploy       Deploy the complete MLOps stack"
    echo "  delete       Delete the complete MLOps stack"
    echo "  status       Show status of deployed resources"
    echo "  logs         Show logs from all pods"
    echo "  port-forward Set up port forwarding for local access"
    echo "  help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 deploy        # Deploy the stack"
    echo "  $0 status        # Check deployment status"
    echo "  $0 port-forward  # Access services locally"
    echo "  $0 delete        # Clean up everything"
}

# Main script logic
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    delete)
        delete
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    port-forward)
        port_forward
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
