#!/bin/bash

# Script para buildar APK de release e fazer upload para Firebase App Distribution
# Uso: ./scripts/build_and_distribute.sh [RELEASE_NOTES]

set -e

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configurações
APP_ID="1:937764727202:android:efbb5e2f34b17b269e6afb"
APK_PATH="build/app/outputs/flutter-apk/app-release.apk"

# Release notes (pode ser passado como argumento ou usa padrão)
RELEASE_NOTES="${1:-Nova versão de teste - $(date '+%d/%m/%Y %H:%M')}"

echo -e "${YELLOW}🔨 Buildando APK de release...${NC}"
flutter build apk --release

if [ ! -f "$APK_PATH" ]; then
    echo -e "${RED}❌ APK não encontrado em: $APK_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✅ APK buildado com sucesso!${NC}"
echo -e "${YELLOW}📤 Fazendo upload para Firebase App Distribution...${NC}"

# Upload para Firebase App Distribution (sem grupo - adicione testers manualmente no console)
firebase appdistribution:distribute "$APK_PATH" \
    --app "$APP_ID" \
    --release-notes "$RELEASE_NOTES" --groups "beta"

echo -e "${GREEN}✅ Upload concluído!${NC}"
echo -e "${YELLOW}📋 Próximos passos:${NC}"
echo -e "   1. Acesse: https://console.firebase.google.com/project/famachappp/appdistribution${NC}"
echo -e "   2. Adicione testers manualmente na seção 'Testers & Groups'${NC}"
echo -e "   3. Envie o link de convite para os testers${NC}"
echo -e "${GREEN}📱 APK disponível no Firebase App Distribution${NC}"