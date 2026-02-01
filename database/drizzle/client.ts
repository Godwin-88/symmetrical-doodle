/**
 * Drizzle ORM Database Client
 *
 * Provides database connection and query builder for the trading system.
 *
 * Usage:
 *   import { db } from './client';
 *   const assets = await db.select().from(assets);
 */

import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from './schema';

// Get the database URL from environment
const connectionString = process.env.DATABASE_URL;

if (!connectionString) {
  console.warn('DATABASE_URL not set - database client will not be initialized');
}

// Create the postgres client
const client = connectionString
  ? postgres(connectionString, {
      max: 10, // Maximum number of connections
      idle_timeout: 20, // Max seconds a client can be idle before being closed
      connect_timeout: 10, // Max seconds to wait for connection
    })
  : null;

// Create the drizzle client
export const db = client
  ? drizzle(client, { schema })
  : null;

// Export schema for convenience
export * from './schema';

// Type for the database instance
export type Database = NonNullable<typeof db>;

/**
 * Check if the database is connected
 */
export async function checkConnection(): Promise<boolean> {
  if (!client) return false;
  try {
    await client`SELECT 1`;
    return true;
  } catch (error) {
    console.error('Database connection check failed:', error);
    return false;
  }
}

/**
 * Close the database connection
 */
export async function closeConnection(): Promise<void> {
  if (client) {
    await client.end();
  }
}

/**
 * Execute a raw SQL query
 */
export async function raw<T = unknown>(sql: string, params?: unknown[]): Promise<T[]> {
  if (!client) throw new Error('Database not connected');
  return client.unsafe(sql, params as never[]) as Promise<T[]>;
}
