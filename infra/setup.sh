#!/bin/bash
# metis cloud infrastructure setup
# run once to create all azure resources

set -e

RESOURCE_GROUP="metis"
LOCATION="westeurope"
SEARCH_NAME="metis-search"
STORAGE_NAME="metisstore42"  # no hyphens allowed in storage names
FUNCTION_APP="metis-automata"
OPENAI_NAME="metis-openai"

echo "registering resource providers..."
az provider register --namespace Microsoft.Search --wait
az provider register --namespace Microsoft.Storage --wait
az provider register --namespace Microsoft.CognitiveServices --wait
az provider register --namespace Microsoft.Web --wait

echo "creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

echo "creating azure AI search (basic tier)..."
az search service create \
  --name $SEARCH_NAME \
  --resource-group $RESOURCE_GROUP \
  --sku basic \
  --location $LOCATION

echo "creating storage account..."
az storage account create \
  --name $STORAGE_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

echo "creating blob containers..."
az storage container create \
  --name ingest \
  --account-name $STORAGE_NAME

az storage container create \
  --name pending \
  --account-name $STORAGE_NAME

# azure openai: skipped (using regular openai until quota approved)
# uncomment when quota is available:
# echo "creating azure openai resource..."
# az cognitiveservices account create \
#   --name $OPENAI_NAME \
#   --resource-group $RESOURCE_GROUP \
#   --location $LOCATION \
#   --kind OpenAI \
#   --sku S0
#
# echo "deploying openai models..."
# az cognitiveservices account deployment create \
#   --name $OPENAI_NAME \
#   --resource-group $RESOURCE_GROUP \
#   --deployment-name gpt-4o \
#   --model-name gpt-4o \
#   --model-version "2024-11-20" \
#   --model-format OpenAI \
#   --sku-capacity 10 \
#   --sku-name GlobalStandard
#
# az cognitiveservices account deployment create \
#   --name $OPENAI_NAME \
#   --resource-group $RESOURCE_GROUP \
#   --deployment-name text-embedding-3-small \
#   --model-name text-embedding-3-small \
#   --model-version "1" \
#   --model-format OpenAI \
#   --sku-capacity 10 \
#   --sku-name GlobalStandard

echo "creating function app..."
az functionapp create \
  --name $FUNCTION_APP \
  --resource-group $RESOURCE_GROUP \
  --storage-account $STORAGE_NAME \
  --consumption-plan-location $LOCATION \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --os-type Linux

echo ""
echo "done. fetching keys..."
echo ""

echo "search admin key:"
az search admin-key show \
  --service-name $SEARCH_NAME \
  --resource-group $RESOURCE_GROUP \
  --query primaryKey -o tsv

echo ""
echo "storage connection string:"
az storage account show-connection-string \
  --name $STORAGE_NAME \
  --resource-group $RESOURCE_GROUP \
  --query connectionString -o tsv

echo ""
echo "search endpoint: https://$SEARCH_NAME.search.windows.net"
echo ""
echo "using regular openai (existing key). switch to azure openai when quota approved."
echo ""
echo "save these keys. run:"
echo "  metis secret set azure-search-endpoint"
echo "  metis secret set azure-search-key"
