<template>
  <div class="flex h-full flex-col gap-6 p-6">
    <!-- API Keys Section -->
    <div class="flex flex-col gap-4">
      <div class="border-b border-gray-600 pb-2 text-lg font-bold">
        {{ $t('setting.apiKey') }}
      </div>
      <div class="flex flex-col gap-4">
        <div
          v-for="key in ['civitai', 'huggingface', 'modelscope']"
          :key="key"
          class="flex flex-col gap-2"
        >
          <div class="text-sm capitalize opacity-70">{{ key }} API Key</div>
          <div
            class="flex items-center gap-4 rounded-lg border border-gray-700 bg-gray-800 p-3"
          >
            <div class="flex-1 truncate font-mono text-sm">
              {{ apiKeyInfo[key] || 'None' }}
            </div>
            <div class="flex gap-2">
              <Button
                icon="pi pi-pencil"
                severity="secondary"
                text
                rounded
                @click="editKey(key)"
              />
              <Button
                v-if="apiKeyInfo[key]"
                icon="pi pi-trash"
                severity="danger"
                text
                rounded
                @click="removeKey(key)"
              />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Other Settings can be added here -->
  </div>
</template>

<script setup lang="ts">
import SettingApiKey from 'components/SettingApiKey.vue'
import { useConfig } from 'hooks/config'
import { useDialog } from 'hooks/dialog'
import { request } from 'hooks/request'
import { useToast } from 'hooks/toast'
import Button from 'primevue/button'
import { useI18n } from 'vue-i18n'

const { apiKeyInfo } = useConfig()
const { t } = useI18n()
const dialog = useDialog()
const { confirm, toast } = useToast()

const editKey = (key: string) => {
  dialog.open({
    key: `setting.api_key.${key}`,
    title: t(`setting.api_key.${key}`),
    content: SettingApiKey,
    modal: true,
    defaultSize: {
      width: 500,
      height: 200,
    },
    contentProps: {
      keyField: key,
      setter: (val: string) => {
        apiKeyInfo.value[key] = val
      },
    },
  })
}

const removeKey = async (key: string) => {
  const accepted = await new Promise<boolean>((resolve) => {
    confirm.require({
      message: t('deleteAsk'),
      header: 'Danger',
      icon: 'pi pi-info-circle',
      accept: () => resolve(true),
      reject: () => resolve(false),
    })
  })
  if (!accepted) {
    return
  }

  try {
    await request('/download/setting', {
      method: 'POST',
      body: JSON.stringify({ key, value: null }),
    })
    apiKeyInfo.value[key] = ''
  } catch (error) {
    toast.add({
      severity: 'error',
      summary: 'Error',
      detail: error.message,
      life: 3000,
    })
  }
}
</script>
